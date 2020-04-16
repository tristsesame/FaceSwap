#define XTENSOR_USE_XSIMD 1
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xshape.hpp"
#include "xtensor/xiterable.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor/xassign.hpp"
#include "xtensor/xlayout.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor-blas/xlinalg.hpp"

using namespace xt::placeholders;

#include <algorithm>
#include <iostream>
#include <vector>
#include <time.h>
#include <windows.h>
#include <tchar.h>
#include <commctrl.h>
#include <windowsx.h>
#include <omp.h>
#include <gl/gl.h>
#include <gl/glu.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/cvdef.h>

using namespace std;
using namespace cv;
using namespace dlib;

#define BUFFER 1024

string winname = "Face Swap";
static int nParams = 20;
HDC hDC;
HGLRC hRC;

xt::xarray<double> arr_mean3DShape;
xt::xarray<int> arr_mesh;
xt::xarray<int> arr_idxs3D;
xt::xarray<int> arr_idxs2D;
xt::xarray<double> arr_blendshapes;
xt::xarray<double> arr_textureCoords;
xt::xarray<double> arr_slice_mean3DShape;
xt::xarray<double> arr_slice_blendshapes;

xt::xarray<double> myRodrigues
(
    xt::xarray<double>& r
)
{
    cv::Mat r1( 1, 3, CV_64FC1, r.data() );
    cv::Mat R1;
    cv::Rodrigues( r1, R1 );
    std::vector<double> v;
    for( int i = 0; i < R1.rows; ++i )
    {
        for( int j = 0; j < R1.cols; ++j )
        {
            v.push_back( R1.at<double>(i,j) );
        }
    }
    std::vector<size_t> sh = {3, v.size()/3};
    xt::xarray<double> R = xt::adapt(&(v[0]), v.size(), false, sh);
    
    return R;
}

xt::xarray<double> fun
(
    xt::xarray<double>& params,
    xt::xarray<double>& slice_mean3DShape, // x
    xt::xarray<double>& slice_blendshapes, // x
    xt::xarray<double>& shape2D1
)
{
    xt::xarray<double> s = params[0];
    xt::xarray<double> r = xt::view(params, xt::range(1, 4));
    xt::xarray<double> t = xt::view(params, xt::range(4, 6));
    xt::xarray<double> w = xt::view(params, xt::range(6, _));

    xt::xarray<double> R = myRodrigues(r);
    xt::xarray<double> P = xt::view(R, xt::range(_, 2));
    xt::xarray<double> W = xt::view(w, xt::all(), xt::newaxis(), xt::newaxis());
    xt::xarray<double> W1 = W * slice_blendshapes;
    xt::xarray<double> shape3D = slice_mean3DShape + xt::sum(W1, 0, xt::evaluation_strategy::immediate);
    xt::xarray<double> T = xt::view(t, xt::all(), xt::newaxis());
    
    xt::xarray<double> projected = s * xt::linalg::dot(P, shape3D) + T;

    return projected;
}

void drawPoints
(
    cv::Mat& img, 
    xt::xarray<double>& points,
    Scalar& color = Scalar(0, 255, 0)
)
{
    for(size_t i=0;i<points.shape(0);i++)
    {
        cv::circle(img, Point(points(i,0), points(i,1)), 2, color);
    }
}

void drawCross
(
    cv::Mat& img, 
    xt::xarray<double>& params, 
    cv::Point& center = cv::Point(100, 100),
    double scale = 30.0
)
{
    xt::xarray<double> vparams = xt::view(params,xt::range(1,4));
    xt::xarray<double> R = myRodrigues(vparams);
    xt::xarray<double> points = xt::xarray<double>({{1, 0, 0}, {0, -1, 0}, {0, 0, -1}});
    xt::xarray<double> RT = xt::transpose(R, {1, 0});
    xt::xarray<double> points2D = xt::view(xt::linalg::dot(points, RT), xt::all(), xt::range(_,2));
    xt::xarray<int> c({center.x,center.y});
    points2D = points2D * scale + c;
    
    cv::line(img, center, cv::Point(points2D(0, 0), points2D(0, 1)), cv::Scalar(255, 0, 0), 3);
    cv::line(img, center, cv::Point(points2D(1, 0), points2D(1, 1)), cv::Scalar(0, 255, 0), 3);
    cv::line(img, center, cv::Point(points2D(2, 0), points2D(2, 1)), cv::Scalar(0, 0, 255), 3);
}

void drawMesh
(
    cv::Mat& img, 
    xt::xarray<double>& shape, 
    xt::xarray<int>& mesh, 
    Scalar& color = Scalar(255, 0, 0)
)
{
    for( size_t i = 0; i < mesh.shape(0); ++i )
    {
        xt::xarray<int> point1 = xt::row(shape, mesh(i,0));
        xt::xarray<int> point2 = xt::row(shape, mesh(i,1));
        xt::xarray<int> point3 = xt::row(shape, mesh(i,2));

        cv::line(img, cv::Point(point1[0], point1[1]), cv::Point(point2[0], point2[1]), Scalar(255, 0, 0), 1);
        cv::line(img, cv::Point(point2[0], point2[1]), cv::Point(point3[0], point3[1]), Scalar(255, 0, 0), 1);
        cv::line(img, cv::Point(point3[0], point3[1]), cv::Point(point1[0], point1[1]), Scalar(255, 0, 0), 1);
    }
}

void drawProjectedShape
(
    cv::Mat& img, 
    xt::xarray<double>& slice_mean3DShape,
    xt::xarray<double>& slice_blendshapes,
    xt::xarray<int>& mesh, 
    xt::xarray<double>& params, 
    bool lockedTranslation = false
)
{
    xt::xarray<double> localParams(params);
    xt::xarray<double> tmp;

    if( lockedTranslation )
    {
        localParams[4] = 100;
        localParams[5] = 200;
    }

    xt::xarray<double> projectedShape = fun(localParams, slice_mean3DShape, slice_blendshapes, tmp);
    xt::xarray<double> projectedShapeT = xt::transpose(projectedShape, {1, 0});

    drawPoints(img, projectedShapeT, Scalar(0, 0, 255));
    drawMesh(img, projectedShapeT, mesh);
    drawCross(img, params);
}

static void getFaceKeypoints( Mat& img, frontal_face_detector detector, shape_predictor sp, std::vector< std::vector<Point2f> >& shapes2D )
{
    float imgScale = 1;
    float maxImgSizeForDetection = 640;
    float maxSize = max(img.cols, img.rows);
    Mat scaledImg(img);
    if( maxSize > maxImgSizeForDetection )
    {
        imgScale = maxImgSizeForDetection / maxSize;
        resize(img, scaledImg, Size(), imgScale, imgScale);
    }

    IplImage ipl_img = cvIplImage(scaledImg);
    dlib::cv_image<dlib::bgr_pixel> img1(&ipl_img);

    std::vector<dlib::rectangle> dets1 = detector(img1);

    if( dets1.size() <= 0 )
        return;
      
    for( int j = 0; j < dets1.size(); ++j )
    {
        dlib::rectangle rect(int(dets1[j].left() / imgScale),
                int(dets1[j].top() / imgScale),
                int(dets1[j].right() / imgScale),
                int(dets1[j].bottom() / imgScale));
                
        full_object_detection dlibShape = sp(img1, rect);
        std::vector<Point2f> landmark;
        for (int i = 0; i < dlibShape.num_parts(); ++i)
        {
            float x=dlibShape.part(i).x();
            float y=dlibShape.part(i).y();
            landmark.push_back(Point2f(x,y));
        }
        shapes2D.push_back( landmark );
    }
}

GLuint addTexture
(
    const char* image_name,
    xt::xarray<double>& textureCoords
)
{
    cv::Mat img = cv::imread(image_name);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    GLuint textureId = 100;
    glGenTextures(1, &textureId);

    glBindTexture(GL_TEXTURE_2D, textureId);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.ptr());

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    xt::view( textureCoords, 0, xt::all() ) /= img.cols;
    xt::view( textureCoords, 1, xt::all() ) /= img.rows;
    
    glBindTexture(GL_TEXTURE_2D, 0);

    return textureId;
}

void drawFace
(
    xt::xarray<float>& vertices,
    xt::xarray<int>& mesh,
    xt::xarray<double>& textureCoords,
    GLuint faceTexId
)
{
    std::vector<float> data;
#pragma omp parallel for
    for( int i = 0; i < mesh.shape(0); ++i )
    {
#pragma omp critical
{
        xt::xarray<size_t> row = xt::row(mesh, i);
        for( int j = 0; j < row.size(); ++j )
        {
            xt::xarray<float> texData = xt::view( textureCoords, xt::all(), row[j] );
            xt::xarray<float> vexData = xt::view( vertices, xt::all(), row[j] );
            data.push_back( texData.data()[0] );
            data.push_back( texData.data()[1] );
            data.push_back( texData.data()[2] );
            data.push_back( vexData.data()[0] );
            data.push_back( vexData.data()[1] );
            data.push_back( vexData.data()[2] );
        }
}
    }

#pragma omp barrier
    
    glBindTexture(GL_TEXTURE_2D, faceTexId);
    glBegin(GL_TRIANGLES);
    for( int i = 0; i < data.size(); i+=6 )
    {
        glTexCoord2fv( &(data[i]) );
        glVertex3fv( &(data[i+3]) );
    }
    glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void myRender
(
    size_t w,
    size_t h,
    xt::xarray<float>& vertices,
    xt::xarray<int>& mesh,
    xt::xarray<double>& texCoords,
    GLuint faceTexId,
    xt::xarray<unsigned char>* renderedImg
)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    drawFace(vertices, mesh, texCoords, faceTexId);
    
    cv::Mat img(h, w, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    
    size_t size = img.total();
    int channels = img.channels();
    std::vector<int> shape = {img.rows, img.cols, channels};
    *renderedImg = xt::adapt(img.data, size * channels, xt::no_ownership(), shape);
}

void SetupPixelFormat()
{
    PIXELFORMATDESCRIPTOR pfd;
    ZeroMemory( &pfd, sizeof(pfd));
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    int pfIdx = ChoosePixelFormat(hDC, &pfd);
    if (pfIdx != 0) SetPixelFormat(hDC, pfIdx, &pfd);
}

void InitGraphics( HWND hWnd, int width, int height )
{
    hDC = GetDC(hWnd);
    SetupPixelFormat();
    hRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hRC);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1000, 1000);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void ResizeGraphics()
{
    RECT rw;
    HWND cvWnd = (HWND)cvGetWindowHandle(winname.c_str());
    GetClientRect(cvWnd, &rw);
    
    int width = rw.right - rw.left;
    int height = rw.bottom - rw.top;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1000, 1000);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

bool cvWin( int flag, int width, int height )
{
    cv::namedWindow(winname.c_str(), flag );
    HWND cvWnd = (HWND)cvGetWindowHandle(winname.c_str());
    if (!cvWnd) return false;
    InitGraphics( cvWnd, width, height );

    return true;
}

void jacobian
(
    xt::xarray<double>& params, 
    xt::xarray<double>& slice_mean3DShape,
    xt::xarray<double>& slice_blendshapes,
    xt::xarray<double>& shape2D1,
    xt::xarray<double>* jacobian1
)
{
    xt::xarray<double> s = params[0];
    xt::xarray<double> r = xt::view(params, xt::range(1, 4));
    xt::xarray<double> t = xt::view(params, xt::range(4, 6));
    xt::xarray<double> w = xt::view(params, xt::range(6, _));

    xt::xarray<double> R = myRodrigues(r);
    xt::xarray<double> P = xt::view(R, xt::range(_, 2));
    xt::xarray<double> W = xt::view(w, xt::all(), xt::newaxis(), xt::newaxis());
    xt::xarray<double> W1 = W * slice_blendshapes;
    xt::xarray<double> shape3D = slice_mean3DShape + xt::sum(W1, 0,xt::evaluation_strategy::immediate);

    int nPoints = slice_mean3DShape.shape(1);

    *jacobian1 = xt::zeros<double>({nPoints * 2, nParams});
    xt::view(*jacobian1, xt::all(), 0) = xt::flatten( xt::linalg::dot(P, shape3D) );

    double stepSize = 10e-4;
    xt::xarray<double> step = xt::zeros<double>({nParams});
    
#pragma omp parallel for    
    for( int i = 1; i < 4; i++ )
    {
#pragma omp critical
{
        step = xt::zeros<double>({nParams});
        xt::view(step, i, xt::all()) = stepSize;
        xt::xarray<double> param1 = params + step;
        xt::xarray<double> p1 = (fun(param1, slice_mean3DShape, slice_blendshapes, shape2D1) - fun(params, slice_mean3DShape, slice_blendshapes, shape2D1)) / stepSize;
        xt::view(*jacobian1, xt::all(), i) = xt::flatten( p1 );
}
    }
    xt::view(*jacobian1, xt::range(_, nPoints), 4) = 1;
    xt::view(*jacobian1, xt::range(nPoints, _), 5) = 1;
    int nBlendshapes = nParams - 6;
    int startIdx = nParams - nBlendshapes;
    
#pragma omp parallel for
    for( int i = 0; i < nBlendshapes; ++i )
    {
#pragma omp critical
{
        xt::view(*jacobian1, xt::all(), i + startIdx) = s * xt::flatten( xt::linalg::dot(P, xt::view(slice_blendshapes, i, xt::all())) );
}
    }
    return;
}

xt::xarray<double> residual
(
    xt::xarray<double>& params, 
    xt::xarray<double>& slice_mean3DShape,
    xt::xarray<double>& slice_blendshapes,
    xt::xarray<double>& shape2D1
)
{
    xt::xarray<double> r = shape2D1 - fun( params, slice_mean3DShape, slice_blendshapes, shape2D1 );
    return xt::flatten(r);
}

xt::xarray<double> LineSearchFun
(
    xt::xarray<double>& alpha,
    xt::xarray<double>& x, // params
    xt::xarray<double>& d, // direction
    xt::xarray<double>& slice_mean3DShape, // x
    xt::xarray<double>& slice_blendshapes, // x
    xt::xarray<double>& shape2D1 // y
)
{
    xt::xarray<double> x0 = x + alpha * d;
    xt::xarray<double> r = residual(x0, slice_mean3DShape, slice_blendshapes, shape2D1 );
    xt::xarray<double> R = r * r;
    return xt::sum(R, xt::evaluation_strategy::immediate);
}

xt::xarray<double> GaussNewton
(
    xt::xarray<double>& x0, 
    xt::xarray<double>& slice_mean3DShape,
    xt::xarray<double>& slice_blendshapes,
    xt::xarray<double>& shape2D1,
    int maxIter=1,
    double eps=10e-7,
    int verbose=1
)
{
    int oldCost = -1;
    xt::xarray<double> x = xt::cast<double>(x0);
    for( int i = 0; i < maxIter; ++i )
    {
        xt::xarray<double> r = residual(x, slice_mean3DShape, slice_blendshapes, shape2D1);
        double cost = xt::sum(r * r, xt::evaluation_strategy::immediate)[0];

        if (cost < eps || abs(cost - oldCost) < eps)
            break;
        oldCost = cost;
        
        xt::xarray<double> J;
        jacobian(x, slice_mean3DShape, slice_blendshapes, shape2D1, &J);
        xt::xarray<double> JT = xt::transpose(J, {1, 0});
        xt::xarray<double> direction = xt::linalg::solve(xt::linalg::dot(JT, J), xt::linalg::dot(JT, r));

        x = x + direction;
    }
    
    return x;
}

void colorTransfer
( 
    xt::xarray<unsigned char>& src,
    xt::xarray<unsigned char>& dst, 
    xt::xarray<unsigned char>& mask,
    xt::xarray<unsigned char>* transferredDst
)
{
    *transferredDst = dst;

    std::vector<std::vector<size_t>> maskIndices = xt::nonzero(mask);
    
    xt::xarray<int> maskedSrc;
    xt::xarray<int> meanSrc;
    xt::xarray<int> maskedDst;
    xt::xarray<int> meanDst;

#pragma omp parallel sections
{
#pragma omp section
{
    std::vector< unsigned char > stripSrc;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( src, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripSrc.push_back( v[j] );
        }
    }
    std::vector<size_t> sh = {stripSrc.size()/3,3};
    maskedSrc = xt::adapt(&(stripSrc[0]), stripSrc.size(), false, sh);

    meanSrc = xt::mean(maskedSrc, 0, xt::evaluation_strategy::immediate);
}
#pragma omp section
{
    // dst
    std::vector< unsigned char > stripDst;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( dst, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripDst.push_back( v[j] );
        }
    }
    std::vector<size_t> dsh = {stripDst.size()/3,3};
    maskedDst = xt::adapt(&(stripDst[0]), stripDst.size(), false, dsh);

    meanDst = xt::mean(maskedDst, 0, xt::evaluation_strategy::immediate);
}
}

#pragma omp barrier

    maskedDst = maskedDst - meanDst;
    maskedDst = maskedDst + meanSrc;
    maskedDst = xt::clip(maskedDst, 0, 255);
    
#pragma omp parallel for
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
#pragma omp critical
{
        xt::view( *transferredDst, maskIndices[0][i], maskIndices[1][i], xt::all() ) = xt::row(maskedDst, i);
}
    }
}

void blendImages
(
    xt::xarray<unsigned char>& src,
    xt::xarray<unsigned char>& dst,
    xt::xarray<unsigned char>& mask,
    double featherAmount,
    xt::xarray<unsigned char>* composedImg
)
{
    *composedImg = xt::cast<unsigned char>(dst);
    std::vector<std::vector<std::size_t>> maskIndices = xt::nonzero(mask);

    std::vector<int> vmask;
#pragma omp parallel for
    for( int i = 0; i < maskIndices.size(); ++i )
    {
#pragma omp critical
{
        for( int j = 0; j < maskIndices[i].size(); ++j )
        {
            vmask.push_back( maskIndices[i][j] );
        }
}
    }
    std::vector<size_t> sh1 = {maskIndices.size(), maskIndices[0].size()};
    xt::xarray<int> xmask = xt::adapt(&(vmask[0]), vmask.size(), false, sh1);
    
    xt::xarray<unsigned char> wsrc;
    xt::xarray<unsigned char> wdst;
    xt::xarray<float> weight;
    
#pragma omp parallel sections
{
#pragma omp section
{
    xt::xarray<float> fmaskPts = xt::hstack(xtuple(xt::view(xt::row(xmask, 1), xt::all(), xt::newaxis()), xt::view(xt::row(xmask, 0), xt::all(), xt::newaxis())));
    xt::xarray<float> faceSize = xt::amax(fmaskPts, 0) - xt::amin(fmaskPts, 0);
    xt::xarray<float> featherAmount1 = featherAmount * xt::amax(faceSize);
    featherAmount = featherAmount1[0];

    cv::Mat cv_maskPts( fmaskPts.shape(0), fmaskPts.shape(1), CV_32FC1, fmaskPts.data() );
    std::vector< Point2f >hull;
    cv::convexHull(cv_maskPts, hull);

    xt::xarray<float> dists = xt::zeros<float>({fmaskPts.shape(0)});
    for( int i = 0; i < fmaskPts.shape(0); ++i )
        dists[i] = cv::pointPolygonTest( hull, Point2f(fmaskPts(i, 0), fmaskPts(i, 1)), true );

    weight = xt::view( xt::clip(dists / featherAmount, 0, 1), xt::all(), xt::newaxis());
}
#pragma omp section
{
    std::vector< unsigned char > stripSrc;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( src, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripSrc.push_back( v[j] );
        }
    }
    std::vector<size_t> sh3 = {stripSrc.size()/3,3};
    wsrc = xt::adapt(&(stripSrc[0]), stripSrc.size(), false, sh3);
}
#pragma omp section
{
    std::vector< unsigned char > stripDst;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( dst, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripDst.push_back( v[j] );
        }
    }
    std::vector<size_t> sh4 = {stripDst.size()/3,3};
    wdst = xt::adapt(&(stripDst[0]), stripDst.size(), false, sh4);
}
}

#pragma omp barrier

    xt::xarray<float> msrc = weight * wsrc;
    xt::xarray<float> mdst = (1-weight) * wdst;

#pragma omp parallel for
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
#pragma omp critical
{
        xt::view( *composedImg, maskIndices[0][i], maskIndices[1][i], xt::all() ) = xt::row(msrc, i) + xt::row(mdst, i);
}
    }
}

void blendImages1
(
    xt::xarray<unsigned char>& rData,
    xt::xarray<unsigned char>& capData,
    xt::xarray<unsigned char>& mask,
    double featherAmount,
    xt::xarray<unsigned char>* transferredDst,
    xt::xarray<unsigned char>* composedImg
)
{
    *composedImg = capData;
    *transferredDst = rData;
    std::vector<std::vector<std::size_t>> maskIndices = xt::nonzero(mask);
    std::vector<int> vmask;

#pragma omp parallel for
    for( int i = 0; i < maskIndices.size(); ++i )
    {
#pragma omp critical
{
        for( int j = 0; j < maskIndices[i].size(); ++j )
        {
            vmask.push_back( maskIndices[i][j] );
        }
}
    }
    std::vector<size_t> sh1 = {maskIndices.size(), maskIndices[0].size()};
    xt::xarray<int> xmask = xt::adapt(&(vmask[0]), vmask.size(), false, sh1);
    
    xt::xarray<unsigned char> bsrc, csrc;
    xt::xarray<unsigned char> bdst;
    xt::xarray<int> maskedSrc;
    xt::xarray<int> meanSrc;
    xt::xarray<int> maskedDst;
    xt::xarray<int> meanDst;
    xt::xarray<float> weight;
    
#pragma omp parallel sections
{
#pragma omp section
{
    std::vector< unsigned char > stripSrc;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( capData, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripSrc.push_back( v[j] );
        }
    }
    std::vector<size_t> sh3 = {stripSrc.size()/3,3};
    maskedSrc = xt::adapt(&(stripSrc[0]), stripSrc.size(), false, sh3);
    csrc = xt::adapt(&(stripSrc[0]), stripSrc.size(), false, sh3);
    meanSrc = xt::mean(maskedSrc, 0, xt::evaluation_strategy::immediate);
}
#pragma omp section
{
    std::vector< unsigned char > stripDst;
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::xarray<unsigned char> v = xt::view( rData, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripDst.push_back( v[j] );
        }
    }
    std::vector<size_t> sh4 = {stripDst.size()/3,3};
    maskedDst = xt::adapt(&(stripDst[0]), stripDst.size(), false, sh4);
    meanDst = xt::mean(maskedDst, 0, xt::evaluation_strategy::immediate);
}
#pragma omp section
{
    xt::xarray<float> fmaskPts = xt::hstack(xtuple(xt::view(xt::row(xmask, 1), xt::all(), xt::newaxis()), xt::view(xt::row(xmask, 0), xt::all(), xt::newaxis())));
    xt::xarray<float> faceSize = xt::amax(fmaskPts, 0) - xt::amin(fmaskPts, 0);
    xt::xarray<float> featherAmount1 = featherAmount * xt::amax(faceSize);
    featherAmount = featherAmount1[0];

    cv::Mat cv_maskPts( fmaskPts.shape(0), fmaskPts.shape(1), CV_32FC1, fmaskPts.data() );
    std::vector< Point2f >hull;
    cv::convexHull(cv_maskPts, hull);

    xt::xarray<float> dists = xt::zeros<float>({fmaskPts.shape(0)});
    for( int i = 0; i < fmaskPts.shape(0); ++i )
        dists[i] = cv::pointPolygonTest( hull, Point2f(fmaskPts(i, 0), fmaskPts(i, 1)), true );

    weight = xt::view( xt::clip(dists / featherAmount, 0, 1), xt::all(), xt::newaxis());
}
}

#pragma omp barrier

    maskedDst = maskedDst - meanDst;
    maskedDst = maskedDst + meanSrc;
    maskedDst = xt::clip(maskedDst, 0, 255);
    
#pragma omp parallel for
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::view( *transferredDst, maskIndices[0][i], maskIndices[1][i], xt::all() ) = xt::row(maskedDst, i);
    }

#pragma omp barrier

    std::vector< unsigned char > stripDst;
#pragma omp parallel for ordered
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
#pragma omp ordered
{
        xt::xarray<unsigned char> v = xt::view( *transferredDst, maskIndices[0][i], maskIndices[1][i], xt::all() );
        for( int j = 0; j < 3; j++ )
        {
            stripDst.push_back( v[j] );
        }
}
    }
    std::vector<size_t> sh4 = {stripDst.size()/3,3};
    bsrc = xt::adapt(&(stripDst[0]), stripDst.size(), false, sh4);

    xt::xarray<float> msrc = weight * bsrc;
    xt::xarray<float> mdst = (1-weight) * csrc;

#pragma omp parallel for
    for( int i = 0; i < maskIndices[0].size(); i++ )
    {
        xt::view( *composedImg, maskIndices[0][i], maskIndices[1][i], xt::all() ) = xt::row(msrc, i) + xt::row(mdst, i);
    }
}

void getShape3D
(
    xt::xarray<double>& mean3DShape,
    xt::xarray<double>& blendshapes, 
    xt::xarray<double>& params,
    xt::xarray<float>* shape3D
)
{
    xt::xarray<double> s;
    xt::xarray<double> t;
    xt::xarray<double> R;
    xt::xarray<double> W1;
#pragma omp parallel sections
{
#pragma omp section
{
    s = xt::view(params, 0, xt::all());
    xt::xarray<double> r = xt::view(params, xt::range(1, 4));
    R = myRodrigues(r);
    t = xt::view(params, xt::range(4, 6));
}
#pragma omp section
{
    xt::xarray<double> w = xt::view(params, xt::range(6, _));
    W1 = xt::view(w, xt::all(), xt::newaxis(), xt::newaxis()) * blendshapes;
}
}

#pragma omp barrier

    *shape3D = mean3DShape + xt::sum(W1, 0, xt::evaluation_strategy::immediate);
    *shape3D = s * xt::linalg::dot(R, *shape3D);
    xt::view(*shape3D, xt::range(_, 2), xt::all()) = xt::view(*shape3D, xt::range(_, 2), xt::all()) + xt::view(t, xt::all(), xt::newaxis());
}

xt::xarray<double> getInitialParameters
(
    xt::xarray<double>& slice_mean3DShape,
    xt::xarray<double>& slice_shape
)
{
    xt::xarray<double> mean3DShape = xt::transpose(slice_mean3DShape, {1, 0});
    xt::xarray<double> shape3DCentered = mean3DShape - xt::mean(mean3DShape, 0, xt::evaluation_strategy::immediate); 
    xt::xarray<double> slice_shape3DCentered = xt::view(shape3DCentered, xt::all(), xt::range(_, 2) );
    xt::xarray<double> slice_mean3DShape1 = xt::view(mean3DShape, xt::all(), xt::range(_, 2) );
    
    xt::xarray<double> shape2D = xt::transpose(slice_shape, {1, 0});
    xt::xarray<double> shape2DCentered = shape2D - xt::mean(shape2D, 0, xt::evaluation_strategy::immediate);
    
    double scale = xt::linalg::norm(shape2DCentered) / xt::linalg::norm(slice_shape3DCentered);
    xt::xarray<double> t = xt::mean(shape2D, 0) - xt::mean(slice_mean3DShape1, 0, xt::evaluation_strategy::immediate);

    xt::xarray<double> params = xt::zeros<double>({nParams});
    params[0] = scale;
    params[4] = t[0];
    params[5] = t[1];
    
    return params;
}

void capCam( xt::xarray<double>& mean3DShape,
             xt::xarray<int>& mesh,
             xt::xarray<int>& idxs3D,
             xt::xarray<int>& idxs2D,
             xt::xarray<double>& blendshapes,
             xt::xarray<double>& textureCoords,
             xt::xarray<double>& slice_mean3DShape,
             xt::xarray<double>& slice_blendshapes,
             char* predictor_path,
             char* image_name,
             char* model_path
           )
{
    int frameno = 0;

    //Open the default video camera
    VideoCapture cap(0);

    // if not success, exit program
    if (cap.isOpened() == false)  
    {
        cout << "Cannot open the video camera" << endl;
        cin.get(); //wait for any key press
    }
    
    int dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    int dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    
    int flag = WINDOW_AUTOSIZE; // CV_WINDOW_NORMAL
    cvWin( flag, dWidth, dHeight );
    
    size_t texId = addTexture( image_name, textureCoords );
    
    dlib::frontal_face_detector fd = get_frontal_face_detector();
    shape_predictor sp;
    deserialize(predictor_path) >> sp;
    
    bool restart = false;
    bool drawPoint = false;
    bool drawLine = false;
    while (true)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video 
        
        //Breaking the while loop if the frames cannot be captured
        if (bSuccess == false)
        {
            cout << "Video camera is disconnected" << endl;
            cin.get(); //Wait for any key press
            break;
        }
        
        __int64 start = cv::getTickCount();

        std::vector< std::vector<Point2f> > shapes2D;
        getFaceKeypoints( frame, fd, sp, shapes2D );
        
        size_t size = frame.total();
        int channels = frame.channels();
        std::vector<int> shape = {frame.rows, frame.cols, channels};
        xt::xarray<unsigned char> xframe = xt::adapt(frame.data, size * channels, xt::no_ownership(), shape);

        cv::Mat blend(frame);
        cv::Mat gl = Mat::zeros(dHeight, dWidth, CV_8UC3);
        xt::xarray<unsigned char> bData;
        xt::xarray<unsigned char> rData;
        xt::xarray<unsigned char> tData;
        for( int inx = 0; inx < shapes2D.size(); ++inx )
        {
            xt::static_shape<std::size_t, 1> sh7 = { shapes2D[inx].size() };
            xt::xarray<Point2f> shape2D = xt::adapt(&(shapes2D[inx][0]), shapes2D[inx].size(), false, sh7);

            xt::xarray<Point2f> slice_shape2D = xt::index_view(shape2D, idxs2D);
            std::vector<double> pos;
            for(int i=0;i<slice_shape2D.size();i++) pos.push_back( slice_shape2D[i].x );
            for(int i=0;i<slice_shape2D.size();i++) pos.push_back( slice_shape2D[i].y );
            std::vector<size_t> sh8 = {2, pos.size()/2};
            xt::xarray<double> slice_shape2D1 = xt::adapt(&(pos[0]), pos.size(), false, sh8);

            xt::xarray<double> modelParams = getInitialParameters( slice_mean3DShape, slice_shape2D1 );
            modelParams = GaussNewton(modelParams, slice_mean3DShape, slice_blendshapes, slice_shape2D1 );
            
            xt::xarray<float> shape3D;
            getShape3D(mean3DShape, blendshapes, modelParams, &shape3D);
            
            myRender( dWidth, dHeight, shape3D, mesh, textureCoords, texId, &rData );
            xt::xarray<unsigned char> mask = xt::view( rData, xt::all(), xt::all(), 0 );

            blendImages1( rData, xframe, mask, 0.2, &tData, &bData );
            blend = cv::Mat( dHeight, dWidth, CV_8UC3, bData.data() );

            if( drawPoint )
            {
                std::vector<double> pos0;
                for(size_t i=0;i<shape2D.size();i++)
                {
                    pos0.push_back( shape2D[i].x );
                }
                for(size_t i=0;i<shape2D.size();i++)
                {
                    pos0.push_back( shape2D[i].y );
                }
                xt::static_shape<std::size_t, 1> sh9 = { pos0.size() };
                xt::xarray<double> shape2D1 = xt::adapt(&(pos0[0]), pos0.size(), false, sh9);
                shape2D1.reshape({2, pos0.size()/2});
            
                xt::xarray<double> shape2D1T = xt::transpose(shape2D1, {1, 0});
                drawPoints( blend, shape2D1T );
            }
            if( drawLine )
            {
                drawProjectedShape(blend, mean3DShape, blendshapes, mesh, modelParams);
            }
            gl = cv::Mat( dHeight, dWidth, CV_8UC3, rData.data() );
        }
        
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        //std::cout << "FPS : " << fps << std::endl;
        char fpstext[1024] = "";
        sprintf( fpstext, "%f", fps );
        cv::putText(gl, fpstext, Point(0,30), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar::all(255), 2, 1);
        
        imshow("Face", gl);
        imshow(winname.c_str(), blend);

        //wait for for 10 ms until any key is pressed.  
        //If the 'Esc' key is pressed, break the while loop.
        //If the any other key is pressed, continue the loop 
        //If any key is not pressed withing 10 ms, continue the loop 
        if (waitKey(10) == 27)
        {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
        if (waitKey(10) == 84 || waitKey(1) == 116) // T/t
            drawPoint = !drawPoint;
        if (waitKey(10) == 75 || waitKey(1) == 107) // K/k
            drawLine = !drawLine;
        if (waitKey(10) == 82 || waitKey(1) == 114) // R/r
        {
            cv::VideoWriter writer("output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(blend.cols, blend.rows));
        }
    }
}

int main(int argc, char *argv[])
{
    char cfg_IniName[BUFFER] = "";         
    GetCurrentDirectory (MAX_PATH, cfg_IniName );
    string config = string(cfg_IniName) + string("\\data.ini" );
    TCHAR predictor_path[BUFFER] = _T("");
    TCHAR image_name[BUFFER] = _T("");
    TCHAR model_path[BUFFER] = _T("");
    TCHAR dataGen[BUFFER] = _T("");
    int a = GetPrivateProfileString(_T("DEFAULT"), _T("predictor"), _T(""), predictor_path, BUFFER, config.c_str());
    int b = GetPrivateProfileString(_T("DEFAULT"), _T("image"), _T(""), image_name, BUFFER, config.c_str());
    int c = GetPrivateProfileString(_T("DEFAULT"), _T("model"), _T(""), model_path, BUFFER, config.c_str());
    int d = GetPrivateProfileString(_T("DEFAULT"), _T("dataGen"), _T(""), dataGen, BUFFER, config.c_str());
    
    cout << predictor_path << endl;
    cout << image_name << endl;
    cout << model_path << endl;
    cout << dataGen << endl;
    
    if( !string(dataGen).empty() ) system( dataGen );
    
    int nProcessors = omp_get_max_threads();
    std::cout << nProcessors << std::endl;
    omp_set_num_threads(nProcessors);
    
    std::string line,line1,line2;
    size_t pos1, pos2;
    std::ifstream infile;
    std::stringstream ss;
    std::vector<int> count;
    std::vector<double> darray;
    std::vector<int> iarray;
   
    infile.open("mean3DShape");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss1(line1);
    //cout << ss1.str() << endl;
    while(getline(ss1, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) darray.push_back( std::stod(line) );
    arr_mean3DShape = xt::adapt(&(darray[0]), darray.size(), false, count);
    //std::cout << arr_mean3DShape << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();

    infile.open("mesh");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss2(line1);
    //std::cout << ss2.str() << endl;
    while(getline(ss2, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) iarray.push_back( std::stoi(line) );
    arr_mesh = xt::adapt(&(iarray[0]), iarray.size(), false, count);
    //std::cout << arr_mesh << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("idxs3D");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss3(line1);
    //cout << ss3.str() << endl;
    while(getline(ss3, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) iarray.push_back( std::stoi(line) );
    arr_idxs3D = xt::adapt(&(iarray[0]), iarray.size(), false, count);
    //std::cout << arr_idxs3D << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("idxs2D");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss4(line1);
    //cout << ss4.str() << endl;
    while(getline(ss4, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) iarray.push_back( std::stoi(line) );
    arr_idxs2D = xt::adapt(&(iarray[0]), iarray.size(), false, count);
    //std::cout << arr_idxs2D << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("blendshapes");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss5(line1);
    //cout << ss5.str() << endl;
    while(getline(ss5, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) darray.push_back( std::stod(line) );
    arr_blendshapes = xt::adapt(&(darray[0]), darray.size(), false, count);
    //std::cout << arr_blendshapes << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("textureCoords");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss6(line1);
    //cout << ss6.str() << endl;
    while(getline(ss6, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) darray.push_back( std::stod(line) );
    arr_textureCoords = xt::adapt(&(darray[0]), darray.size(), false, count);
    //std::cout << arr_textureCoords << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("slice_mean3DShape");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss7(line1);
    //cout << ss7.str() << endl;
    while(getline(ss7, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) darray.push_back( std::stod(line) );
    arr_slice_mean3DShape = xt::adapt(&(darray[0]), darray.size(), false, count);
    //std::cout << arr_slice_mean3DShape << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();
    
    infile.open("slice_blendshapes");
    std::getline(infile, line);
    pos1 = line.find("(");
    pos2 = line.find(")");
    if( pos1 != string::npos && pos2 != string::npos) line1 = line.substr(pos1+1,pos2-pos1-1);
    std::stringstream ss8(line1);
    //cout << ss8.str() << endl;
    while(getline(ss8, line2, ',')) count.push_back(std::stoi(line2));
    while(std::getline(infile, line)) darray.push_back( std::stod(line) );
    arr_slice_blendshapes = xt::adapt(&(darray[0]), darray.size(), false, count);
    //std::cout << arr_slice_blendshapes << endl;
    darray.clear();
    iarray.clear();
    count.clear();
    infile.close();

    capCam(
          arr_mean3DShape,
          arr_mesh,
          arr_idxs3D,
          arr_idxs2D,
          arr_blendshapes,
          arr_textureCoords,
          arr_slice_mean3DShape,
          arr_slice_blendshapes,
          predictor_path,
          image_name,
          model_path
          );

    return 0;
}