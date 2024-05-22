#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float score;
    int c;
    int h;
    int w;
} ScoreCXY;

class CenterNet
{
public:
    CenterNet();

    ~CenterNet();

    int NMS(int8_t *heatmap, int8_t *heatmapmax, int &quant_zp, float &quant_scale);

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

private:
    int class_num = 3;
    int input_h = 384;
    int input_w = 1280;

    int object_thresh = 0.5;

    int output_h = 96;
    int output_w = 320;

    int downsample_ratio = 4;

    std::vector<ScoreCXY> keep_heatmap;
};

#endif
