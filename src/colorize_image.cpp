#include "colorize_image.hpp"

#include <opencv2/opencv.hpp>

using namespace idc;

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
    m_img_rgb.resize(m_Xd, m_Xd);             // !!! ADDED THIS !!!

    // convert into lab space
    _set_img_lab_();
    _set_img_lab_mc_();
}

cv::Mat ColorizeImageBase::get_img_gray() const
{
    // Get black and white image
    cv::Mat zero(m_img_l.rows, m_img_l.cols, m_img_l.type());
    std::vector<cv::Mat> channels{m_img_l, zero, zero};
    cv::Mat merged;
    cv::merge(channels, merged);

    cv::Mat gray;
    cv::cvtColor(merged, gray, cv::COLOR_Lab2RGB);
    
    return gray;
}

cv::Mat ColorizeImageBase::get_img_gray_fullres() const
{
    // Get black and white image
    cv::Mat zero(m_img_l_fullres.rows, m_img_l_fullres.cols, m_img_l_fullres.type());
    std::vector<cv::Mat> channels{m_img_l_fullres, zero, zero};
    cv::Mat merged;
    cv::merge(channels, merged);

    cv::Mat gray;
    cv::cvtColor(merged, gray, cv::COLOR_Lab2RGB);
    
    return gray;
}

cv::Mat ColorizeImageBase::get_img_fullres() const
{
    // This assumes m_img_l_fullres, m_output_ab are set.
    // Typically, this means that set_image() and net_forward()
    // have been called.
    // bilinear upsample
    const double zoom_factor = (1, 1. * m_img_l_fullres.width / m_output_ab.width, 1. * m_img_l_fullres.height / m_output_ab.shapeheight)
    cv::Mat output_ab_fullres;
    cv::resize(m_output_ab, output_ab_fullres, cv::Size(), zoom_factor, zoom_factor);

    std::vector<cv::Mat> ab;
    cv::split(output_ab_fullres, ab);

    std::vector<cv::Mat> img{img_l_fullres, ab[0], ab[1]};
    return cv::merge();
}

void ColorizeImageBase::net_forward(const nc::NdArray<double> &_input_ab, const nc::NdArray<double> &_input_mask)
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