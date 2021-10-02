#pragma once

#include <string>

#include <opencv2/core/mat.hpp>

#define NUMCPP_NO_USE_BOOST
#include <NumCpp.hpp>

namespace idc
{
    class ColorizeImageBase
    {
    private:
        const int m_Xd;
        bool m_img_l_set = false;
        bool m_net_set = false;
        const int m_Xfullres_max;     // maximum size of maximum dimension
        bool m_img_just_set = false;  // this will be true whenever image is just loaded
                                    // net_forward can set this to False if they want

        cv::Mat m_img_rgb_fullres, m_img_rgb;
        cv::Mat m_img_lab_fullres, m_img_l_fullres, m_img_ab_fullres;

        cv::Mat m_img_lab, m_img_l, m_img_ab;
        cv::Mat m_img_lab_mc, m_img_l_mc, m_img_ab_mc;

        cv::Mat m_output_rgb;
        cv::Mat m_output_lab, m_output_ab;

        nc::NdArray<double> m_input_ab;
        nc::NdArray<double> m_input_ab_mc;

        nc::NdArray<double> m_input_mask;
        nc::NdArray<double> m_input_mask_mult;

        const double m_l_norm, m_ab_norm, m_l_mean, m_ab_mean, m_mask_mult;

    public:
        ColorizeImageBase(double l_norm, double ab_norm, double l_mean, double ab_mean, double mask_mult, int Xd=256, int Xfullres_max=10000) 
            : m_l_norm(l_norm)
            , m_ab_norm(ab_norm)
            , m_l_mean(l_mean)
            , m_ab_mean(ab_mean)
            , m_mask_mult(mask_mult)
            , m_Xd(Xd)
            , m_Xfullres_max(Xfullres_max)
        {}

        virtual void prep_net() = 0;

        // ***** Image prepping *****
        void load_image(const std::string &input_path);
        void set_image(const cv::Mat &input_image);

        // NOT IMPLEMENTED
        //get_result_PSNR

        const cv::Mat &get_img_forward() const { return m_output_rgb; }
        cv::Mat get_img_gray() const;
        cv::Mat get_img_gray_fullres() const;
        cv::Mat get_img_fullres() const;

        // NOT IMPLEMENTED
        //get_input_img_fullres
        //get_input_img
        //get_img_mask
        //get_img_mask_fullres
        //get_sup_img
        //get_sup_fullres

    protected:
        // INPUTS
        //  ab       2xXxX    input color patches (non-normalized)
        //  mask     1xXxX    input mask, indicating which points have been provided
        // assumes self.img_l_mc has been set
        void net_forward(const nc::NdArray<double> &input_ab, const nc::NdArray<double> &input_mask);

    private:
        void _set_img_lab_fullres_();
        void _set_img_lab_();
        void _set_img_lab_mc_();
        void _set_img_l_();
        void _set_img_ab_();
        void _set_out_ab_();
    };
}
