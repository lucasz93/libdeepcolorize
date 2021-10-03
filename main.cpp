#include "src/colorize_image.hpp"

static constexpr const char *COLOR_MODEL = "/mechsrc/colorize/interactive-deep-colorization/models/pytorch/caffemodel.pth";

int main(int argc, char **argv)
{
    idc::ColorizeImageTorch colorModel(256, 10000, false);
    colorModel.prep_net(0, COLOR_MODEL, false);
    
    idc::ColorizeImageTorch distModel(256, 10000, false);
    colorModel.prep_net(0, COLOR_MODEL, true);

    return 0;
}