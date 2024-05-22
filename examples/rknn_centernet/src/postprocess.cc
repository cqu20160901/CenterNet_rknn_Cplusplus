#include "postprocess.h"
#include <algorithm>
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = 0;
    float YMin = 0;
    float XMax = 0;
    float YMax = 0;
    float Area1 = 0;
    float Area2 = 0;
    float InterWidth = 0;
    float InterHeight = 0;

    XMin = ZQ_MAX(XMin1, XMin2);
    YMin = ZQ_MAX(YMin1, YMin2);
    XMax = ZQ_MIN(XMax1, XMax2);
    YMax = ZQ_MIN(YMax1, YMax2);

    InterWidth = XMax - XMin;
    InterHeight = YMax - YMin;

    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;

    Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

CenterNet::CenterNet()
{
}

CenterNet::~CenterNet()
{
}

int CenterNet::NMS(int8_t *heatmap, int8_t *heatmapmax, int &quant_zp, float &quant_scale)
{
    keep_heatmap.clear();
    ScoreCXY temp;

    for (int c = 0; c < class_num; c++)
    {
        for (int h = 0; h < output_h; h++)
        {
            for (int w = 0; w < output_w; w++)
            {
                if (heatmapmax[c * output_h * output_w + h * output_w + w] == heatmap[c * output_h * output_w + h * output_w + w] &&
                    heatmap[c * output_h * output_w + h * output_w + w] > object_thresh)
                {
                    temp.score = DeQnt2F32(heatmap[c * output_h * output_w + h * output_w + w], quant_zp, quant_scale);
                    temp.c = c;
                    temp.w = w;
                    temp.h = h;
                    keep_heatmap.push_back(temp);
                }
            }
        }
    }

    return 1;
}

int CenterNet::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
{
    int ret = 0;

    int8_t *heatmap = (int8_t *)pBlob[0];
    int8_t *offset_2d = (int8_t *)pBlob[1];
    int8_t *size_2d = (int8_t *)pBlob[2];
    int8_t *heatmapmax = (int8_t *)pBlob[3];

    int heatmap_zq = qnt_zp[0];
    int offset_2d_zq = qnt_zp[1];
    int size_2d_zq = qnt_zp[2];
    int heatmapmax_zq = qnt_zp[3];

    float heatmap_scale = qnt_scale[0];
    float offset_2d_scale = qnt_scale[1];
    float size_2d_scale = qnt_scale[2];
    float heatmapmax_scale = qnt_scale[3];

    ret = NMS(heatmap, heatmapmax, heatmap_zq, heatmap_scale);
    std::sort(keep_heatmap.begin(), keep_heatmap.end(), [](ScoreCXY &S1, ScoreCXY &S2) -> bool
              { return (S1.score > S2.score); });

    int classId = 0;
    float score = 0;
    int w = 0, h = 0;

    float bx = 0, by = 0, bw = 0, bh = 0;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float xmin1 = 0, ymin1 = 0, xmax1 = 0, ymax1 = 0;
    int keep_flag = 0;
	
    for (int i = 0; i < keep_heatmap.size(); i++)
    {
        if (i > 50)
        {
            break;
        }
        classId = keep_heatmap[i].c;
        score = keep_heatmap[i].score;
        w = keep_heatmap[i].w;
        h = keep_heatmap[i].h;

        bx = (w + DeQnt2F32(offset_2d[0 * output_h * output_w + h * output_w + w], offset_2d_zq, offset_2d_scale)) * downsample_ratio;
        by = (h + DeQnt2F32(offset_2d[1 * output_h * output_w + h * output_w + w], offset_2d_zq, offset_2d_scale)) * downsample_ratio;
        bw = DeQnt2F32(size_2d[0 * output_h * output_w + h * output_w + w], size_2d_zq, size_2d_scale) * downsample_ratio;
        bh = DeQnt2F32(size_2d[1 * output_h * output_w + h * output_w + w], size_2d_zq, size_2d_scale) * downsample_ratio;

        xmin = (bx - bw / 2) / input_w;
        ymin = (by - bh / 2) / input_h;
        xmax = (bx + bw / 2) / input_w;
        ymax = (by + bh / 2) / input_h;

        keep_flag = 0;
        for (int i = 0; i < DetectiontRects.size(); i += 6)
        {
            xmin1 = DetectiontRects[i + 2];
            ymin1 = DetectiontRects[i + 3];
            xmax1 = DetectiontRects[i + 4];
            ymax1 = DetectiontRects[i + 5];

            if (IOU(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.45)
            {
                keep_flag += 1;
                break;
            }
        }

        if (0 == keep_flag)
        {
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin));
            DetectiontRects.push_back(float(ymin));
            DetectiontRects.push_back(float(xmax));
            DetectiontRects.push_back(float(ymax));
        }
    }

    return ret;
}
