#include "colorize_image.hpp"
#include "model.hpp"

#include <opencv2/opencv.hpp>

#define NUMCPP_NO_USE_BOOST
#include <NumCpp.hpp>

using namespace idc;

torch::Tensor mat_to_tensor(const cv::Mat &m)
{
    return torch::from_blob(m.data, {m.rows, m.cols, m.channels()}, torch::kByte);
}

cv::Mat tensor_to_mat(const torch::Tensor &t)
{
    auto tensor = t.squeeze().detach();
    tensor = tensor.permute({1, 2, 0}).contiguous();
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr<uchar>());
    return mat.clone();
}

void ColorizeImageBase::load_image(const std::string &input_path)
{
    // rgb image [CxXdxXd]
    cv::cvtColor(cv::imread(input_path, 1), m_img_rgb_fullres, cv::COLOR_BGR2RGB);
    _set_img_lab_fullres_();

    m_img_rgb_fullres.copyTo(m_img_rgb);
    m_img_rgb.resize(m_Xd, m_Xd);

    m_img_l_set = true;

    // convert into lab space
    _set_img_lab_();
    _set_img_lab_mc_();
}

void ColorizeImageBase::set_image(const cv::Mat &input_image)
{
    input_image.copyTo(m_img_rgb_fullres);
    _set_img_lab_fullres_();

    m_img_l_set = true;

    m_img_rgb_fullres.copyTo(m_img_rgb);

    // convert into lab space
    _set_img_lab_();
    _set_img_lab_mc_();
}

cv::Mat ColorizeImageBase::get_img_gray() const
{
    return _get_img_gray_(m_img_l);
}

cv::Mat ColorizeImageBase::get_img_gray_fullres() const
{
    return _get_img_gray_(m_img_l_fullres);
}

cv::Mat ColorizeImageBase::_get_img_gray_(const cv::Mat &img) const
{
    // Get black and white image
    cv::Mat zero(img.rows, img.cols, img.type());
    std::vector<cv::Mat> channels{img, zero, zero};
    cv::Mat merged;
    cv::merge(channels, merged);

    cv::Mat gray;
    cv::cvtColor(merged, gray, cv::COLOR_Lab2RGB);
    
    return gray;
}

cv::Mat ColorizeImageBase::get_img_fullres() const
{
    return _get_fullres_(m_output_ab);
}

cv::Mat ColorizeImageBase::get_input_img_fullres() const
{
    return _get_fullres_(m_input_ab);
}

cv::Mat ColorizeImageBase::_get_fullres_(const cv::Mat &ab) const
{
    // This assumes m_img_l_fullres, m_output_ab are set.
    // Typically, this means that set_image() and net_forward()
    // have been called.
    // bilinear upsample
    const double zoom_factor = (1, 1. * m_img_l_fullres.rows / ab.rows, 1. * m_img_l_fullres.cols / ab.cols);
    cv::Mat output_ab_fullres;
    cv::resize(ab, output_ab_fullres, cv::Size(), zoom_factor, zoom_factor);

    std::vector<cv::Mat> ab_channels;
    cv::split(output_ab_fullres, ab_channels);

    std::vector<cv::Mat> img_channels{m_img_l_fullres, ab_channels[0], ab_channels[1]};
    cv::Mat img;
    cv::merge(img_channels, img);
    return img;
}

cv::Mat ColorizeImageBase::get_input_img() const
{
    std::vector<cv::Mat> ab_channels;
    cv::split(m_input_ab, ab_channels);

    std::vector<cv::Mat> img_channels{m_img_l, ab_channels[0], ab_channels[1]};
    cv::Mat img;
    cv::merge(img_channels, img);
    return img;
}

cv::Mat ColorizeImageBase::get_img_mask() const
{
    cv::Mat negative;
    cv::bitwise_not(m_input_mask, negative);
    return _get_img_gray_(negative);
}

cv::Mat ColorizeImageBase::get_img_mask_fullres() const
{
    cv::Mat negative;
    cv::bitwise_not(m_input_mask, negative);
    const auto input_mask_fullres = _get_fullres_(negative);

    cv::Mat zero(input_mask_fullres.rows, input_mask_fullres.cols, input_mask_fullres.type());
    std::vector<cv::Mat> channels{input_mask_fullres, zero, zero};
    cv::Mat merged;
    cv::merge(channels, merged);
    return merged;
}

void ColorizeImageBase::net_forward(const cv::Mat &_input_ab, const cv::Mat &_input_mask)
{
    if (!m_img_l_set)
        throw std::logic_error("I need to have an image!");

    if (!m_net_set)
        throw std::logic_error("I need to have a net!");

    m_input_ab = _input_ab;
    m_input_ab_mc = (m_input_ab - m_ab_mean) / m_ab_norm;
    m_input_mask = _input_mask;
    m_input_mask_mult = m_input_mask * m_mask_mult;
}

void ColorizeImageBase::_set_img_lab_fullres_()
{
    // adjust full resolution image to be within maximum dimension is within Xfullres_max
    auto Xfullres = m_img_rgb_fullres.size().width;
    auto Yfullres = m_img_rgb_fullres.size().height;
    if (Xfullres > m_Xfullres_max || Yfullres > m_Xfullres_max)
    {
        double zoom_factor = 0;
        
        if (Xfullres > Yfullres)
            zoom_factor = 1. * m_Xfullres_max / Xfullres;
        else
            zoom_factor = 1. * m_Xfullres_max / Yfullres;

        cv::Mat scaled;
        cv::resize(m_img_rgb_fullres, scaled, cv::Size(), zoom_factor, zoom_factor);
        m_img_rgb_fullres = scaled;
    }

    cv::cvtColor(m_img_rgb_fullres, m_img_lab_fullres, cv::COLOR_RGB2Lab);

    std::vector<cv::Mat> channels;
    cv::split(m_img_lab_fullres, channels);
    m_img_l_fullres = channels[0];
    cv::merge(&channels[1], 2, m_img_ab_fullres);
}

void ColorizeImageBase::_set_img_lab_()
{
    // set m_img_lab from m_im_rgb
    cv::cvtColor(m_img_rgb, m_img_lab, cv::COLOR_RGB2Lab);
    
    std::vector<cv::Mat> channels;
    cv::split(m_img_lab, channels);
    m_img_l = channels[0];
    cv::merge(&channels[1], 2, m_img_ab);
}

void ColorizeImageBase::_set_img_lab_mc_()
{
    // set m_img_lab_mc from m_img_lab
    // lab image, mean centered [XxYxX]
    m_img_lab_mc = m_img_lab / cv::Vec3d{m_l_norm, m_ab_norm, m_ab_norm};
    const cv::Vec3d sub{m_l_mean / m_l_norm, m_ab_mean / m_ab_norm, m_ab_mean / m_ab_norm};
    for (int r = 0; r < m_img_lab_mc.rows; r++)
        m_img_lab_mc.row(r) = m_img_lab_mc.row(r) - sub;
    _set_img_l_();
}

void ColorizeImageBase::_set_img_l_()
{
    std::vector<cv::Mat> channels;
    cv::extractChannel(m_img_lab_mc, m_img_l_mc, 0);
    m_img_l_set = true;
}

void ColorizeImageBase::_set_img_ab_()
{
    std::vector<cv::Mat> channels;
    cv::split(m_img_lab_mc, channels);
    cv::merge(&channels[1], 2, m_img_ab_mc);
}

void ColorizeImageBase::_set_out_ab_()
{
    cv::cvtColor(m_output_lab, m_output_rgb, cv::COLOR_RGB2Lab);

    std::vector<cv::Mat> channels;
    cv::split(m_output_lab, channels);
    cv::merge(&channels[1], 2, m_output_ab);
}

ColorizeImageTorch::ColorizeImageTorch(int Xd, int Xfullres_max, bool maskcent)
    : ColorizeImageBase(1.0, 1.0, 50.0, 0.0, 1.0, Xd, Xfullres_max)
    , m_mask_cent(maskcent ? 0.5 : 0.0)
{
}

void ColorizeImageTorch::prep_net(int gpu_id, const std::string &path, bool dist)
{
    std::cout << "path = " << path << std::endl;
    std::cout << "Model set! dist mode? " << (dist ? "True" : "False") << std::endl;
    m_net = torch::jit::load(path, torch::kCUDA);

    // TODO: How to remove metadata?
    //if (state_dict.hasattr("_metadata"))
    //    del state_dict._metadata;

    /* !!! NOT IMPLEMENTED !!! PyTorch 0.4 is very old at this point. Not supporting dead code.
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        m___patch_instance_norm_state_dict(state_dict, m_net, key.split('.'))
    */

   m_net.to(std::string("cuda:") + std::to_string(gpu_id));
   m_net.eval();
   m_net_set = true;
}

cv::Mat ColorizeImageTorch::net_forward(const cv::Mat &input_ab, const cv::Mat &input_mask)
{
    ColorizeImageBase::net_forward(input_ab, input_mask);

    // TODO: Seems wasteful to convert to tensors here. Could we do away with cv::Mat altogether?
    auto img_l_mc = mat_to_tensor(m_img_l_mc);
    auto input_ab_mc = mat_to_tensor(m_input_ab_mc);
    auto input_mask_mult = mat_to_tensor(m_input_mask_mult);

    auto result = m_net.forward({img_l_mc, input_ab_mc, input_mask_mult, m_mask_cent}).toTensorVector();
    m_output_rgb = tensor_to_mat(result[0]);

    _set_out_ab_();
    return m_output_rgb;
}

ColorizeImageTorchDist::ColorizeImageTorchDist(int Xd, int Xfullres_max, bool maskcent)
    : ColorizeImageTorch(Xd, Xfullres_max, maskcent)
{
    auto pts_grid = nc::meshgrid(nc::arange(-110, 120, 10), nc::arange(-110, 120, 10));
    assert(pts_grid.first.shape().cols == 529 && pts_grid.first.shape().rows == 1);
    assert(pts_grid.second.shape().cols == 529 && pts_grid.second.shape().rows == 1);
    m_in_hull = nc::ones<bool>(pts_grid.first.shape().cols);
    m_AB = pts_grid.first.shape().cols;
    m_A = int(nc::sqrt(m_AB));
    m_B = int(nc::sqrt(m_AB));
    assert(m_A == 23 && m_B == 23);
    m_dist_ab_full = nc::zeros(nc::Shape(m_AB, m_Xd, m_Xd));
    m_dist_ab_grid = nc::zeros(nc::Shape(m_A, m_B, m_Xd, m_Xd));
    m_dist_entropy = nc::zeros(nc::Shape(m_Xd, m_Xd));
}

cv::Mat ColorizeImageTorchDist::net_forward(const cv::Mat &input_ab, const cv::Mat &input_mask)
{
    ColorizeImageBase::net_forward(input_ab, input_mask);

    // TODO: Seems wasteful to convert to tensors here. Could we do away with cv::Mat altogether?
    auto img_l_mc = mat_to_tensor(m_img_l_mc);
    auto input_ab_mc = mat_to_tensor(m_input_ab_mc);
    auto input_mask_mult = mat_to_tensor(m_input_mask_mult);

    // set distribution
    auto result = m_net.forward({img_l_mc, input_ab_mc, input_mask_mult, m_mask_cent}).toTensorVector();
    auto function_return = tensor_to_mat(result[0].data().numpy_T());
    m_dist_ab = tensor_to_mat(result[1].data().numpy_T().data_ptr);
    m_dist_ab_set = true;

    // full grid, ABxXxX, AB = 529
    for (int i = 0; i < m_in_hull.shape().cols; i++)
    {
        if (m_in_hull[i])
            m_dist_ab_full[i, :, :] = m_dist_ab
    }

    // gridded, AxBxXxX, A = 23
    m_dist_ab_grid = m_dist_ab_full.reshape((m_A, m_B, m_Xd, m_Xd))

    return function_return;
}