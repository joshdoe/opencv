/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/******** Prototype of CLAHE function. Put this in a separate include file. *****/
template <typename kz_pixel_t>
void CLAHE(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes, kz_pixel_t Min,
    kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
    unsigned int uiNrBins, float fCliplimit);

void cv::adapthisteq (InputArray _src, OutputArray _dst,
    unsigned int numTilesX, unsigned int numTilesY,
    double clipLimit, unsigned short nBins,
    unsigned short minVal, unsigned short maxVal)
{
    Mat src = _src.getMat ();
    CV_Assert (src.depth() == CV_8U || src.depth() == CV_16U);

    _dst.create (src.size(), CV_8U);
    Mat dst = _dst.getMat ();

    //CLAHE runs in-place
    src.copyTo (dst);

    if (dst.depth () == CV_8U)
        CLAHE<unsigned char> ((unsigned char*) dst.data, dst.cols, dst.rows, minVal, maxVal, numTilesX, numTilesY, nBins, clipLimit);
    else if (dst.depth () == CV_16U)
        CLAHE ((unsigned short*) dst.data, dst.cols, dst.rows, minVal, maxVal, numTilesX, numTilesY, nBins, clipLimit);
}

/*
* ANSI C code from the article
* "Contrast Limited Adaptive Histogram Equalization"
* by Karel Zuiderveld, karel@cv.ruu.nl
* in "Graphics Gems IV", Academic Press, 1994
*
*
*  These functions implement Contrast Limited Adaptive Histogram Equalization.
*  The main routine (CLAHE) expects an input image that is stored contiguously in
*  memory;  the CLAHE output image overwrites the original input image and has the
*  same minimum and maximum values (which must be provided by the user).
*  This implementation assumes that the X- and Y image resolutions are an integer
*  multiple of the X- and Y sizes of the contextual regions. A check on various other
*  error conditions is performed.
*
*  #define the symbol BYTE_IMAGE to make this implementation suitable for
*  8-bit images.
*
*  The code is ANSI-C and is also C++ compliant.
*
*  Author: Karel Zuiderveld, Computer Vision Research Group,
*	     Utrecht, The Netherlands (karel@cv.ruu.nl)
*/

/*********************** Local prototypes ************************/
static void ClipHistogram (unsigned long*, unsigned int, unsigned long);
template <typename kz_pixel_t>
static void MakeHistogram (kz_pixel_t*, unsigned int, unsigned int, unsigned int,
    unsigned long*, unsigned int, kz_pixel_t*);
template <typename kz_pixel_t>
static void MapHistogram (unsigned long*, kz_pixel_t, kz_pixel_t,
    unsigned int, unsigned long);
template <typename kz_pixel_t>
static void MakeLut (kz_pixel_t*, kz_pixel_t, kz_pixel_t, unsigned int);
template <typename kz_pixel_t>
static void Interpolate (kz_pixel_t*, int, unsigned long*, unsigned long*,
    unsigned long*, unsigned long*, unsigned int, unsigned int, kz_pixel_t*);

/**************	 Start of actual code **************/
#include <stdlib.h>			 /* To get prototypes of malloc() and free() */

/************************** main function CLAHE ******************/
template <typename kz_pixel_t>
void CLAHE (kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
    kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
    unsigned int uiNrBins, float fClipLimit)
    /*   pImage - Pointer to the input/output image
    *   uiXRes - Image resolution in the X direction
    *   uiYRes - Image resolution in the Y direction
    *   Min - Minimum greyvalue of input image (also becomes minimum of output image)
    *   Max - Maximum greyvalue of input image (also becomes maximum of output image)
    *   uiNrX - Number of contextial regions in the X direction (min 2)
    *   uiNrY - Number of contextial regions in the Y direction (min 2)
    *   uiNrBins - Number of greybins for histogram ("dynamic range")
    *   float fClipLimit - Normalized cliplimit (higher values give more contrast)
    * The number of "effective" greylevels in the output image is set by uiNrBins; selecting
    * a small value (eg. 128) speeds up processing and still produce an output image of
    * good quality. The output image will have the same minimum and maximum value as the input
    * image. A clip limit smaller than 1 results in standard (non-contrast limited) AHE.
    */
{
    unsigned int uiX, uiY;		  /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned long ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer;		   /* pointer to image */
    kz_pixel_t *aLUT;	    /* lookup table used for scaling of input image */
    unsigned long* pulHist, *pulMapArray; /* pointer to histogram and mappings*/
    unsigned long* pulLU, *pulLB, *pulRU, *pulRB; /* auxiliary pointers interpolation */

    if (uiNrX < 2 || uiNrY < 2)
        CV_Error (CV_StsOutOfRange, "at least 4 contextual regions required");
    if (uiXRes % uiNrX)
        CV_Error (CV_StsBadArg, "width not a multiple of nTilesX");
    if (uiYRes % uiNrY)
        CV_Error (CV_StsBadArg, "height not a multiple of nTilesY");
    if (Max >= (1 << (sizeof (kz_pixel_t) * 8)))
        CV_Error (CV_StsOutOfRange, "maximum cannot exceed highest pixel value");
    if (Min >= Max)
        CV_Error (CV_StsOutOfRange, "minimum cannot exceed maximum");
    if (uiNrBins == 0)
        CV_Error (CV_StsOutOfRange, "nBins must be greater than 0");

    pulMapArray=(unsigned long *)malloc(sizeof(unsigned long)*uiNrX*uiNrY*uiNrBins);
    aLUT = (kz_pixel_t *) malloc (1 << (sizeof (kz_pixel_t) * 8));
    if (!pulMapArray || !aLUT)
        CV_Error (CV_StsNoMem, "Not enough memory! (try reducing nBins)");

    uiXSize = uiXRes/uiNrX; uiYSize = uiYRes/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned long)uiXSize * (unsigned long)uiYSize;

    if (fClipLimit == 1.0)
        return;	  /* is OK, immediately returns original image. FIXME: valid? */
    else if (fClipLimit > 0.0) {		  /* Calculate actual cliplimit	 */
        //FIXME: this doesn't seem to really be normalized 0 to 1
        ulClipLimit = (unsigned long) (fClipLimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else
        ulClipLimit = 1UL<<14;		  /* Large value, do not clip (AHE) */

    MakeLut(aLUT, Min, Max, uiNrBins);	  /* Make lookup table for mapping of greyvalues */

    /* Calculate greylevel mappings for each contextual region */
    for (uiY = 0, pImPointer = pImage; uiY < uiNrY; uiY++) {
        for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize) {
            pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
            MakeHistogram(pImPointer,uiXRes,uiXSize,uiYSize,pulHist,uiNrBins,aLUT);
            ClipHistogram(pulHist, uiNrBins, ulClipLimit);
            MapHistogram(pulHist, Min, Max, uiNrBins, ulNrPixels);
        }
        pImPointer += (uiYSize - 1) * uiXRes;		  /* skip lines, set pointer */
    }

    /* Interpolate greylevel mappings to get CLAHE image */
    for (pImPointer = pImage, uiY = 0; uiY <= uiNrY; uiY++) {
        if (uiY == 0) {					  /* special case: top row */
            uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
        }
        else {
            if (uiY == uiNrY) {				  /* special case: bottom row */
                uiSubY = uiYSize >> 1;	uiYU = uiNrY-1;	 uiYB = uiYU;
            }
            else {					  /* default values */
                uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
            }
        }
        for (uiX = 0; uiX <= uiNrX; uiX++) {
            if (uiX == 0) {				  /* special case: left column */
                uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
            }
            else {
                if (uiX == uiNrX) {			  /* special case: right column */
                    uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
                }
                else {					  /* default values */
                    uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
                }
            }

            pulLU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXL)];
            pulRU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXR)];
            pulLB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXL)];
            pulRB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXR)];
            Interpolate(pImPointer,uiXRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY,aLUT);
            pImPointer += uiSubX;			  /* set pointer on next matrix */
        }
        pImPointer += (uiSubY - 1) * uiXRes;
    }

    free (pulMapArray);
    free (aLUT);
}

void ClipHistogram (unsigned long* pulHistogram, unsigned int
    uiNrGreylevels, unsigned long ulClipLimit)
    /* This function performs clipping of the histogram and redistribution of bins.
    * The histogram is clipped and the number of excess pixels is counted. Afterwards
    * the excess pixels are equally redistributed across the whole histogram (providing
    * the bin count is smaller than the cliplimit).
    */
{
    unsigned long* pulBinPointer, *pulEndPointer, *pulHisto;
    unsigned long ulNrExcess, ulUpper, ulBinIncr, ulStepSize, i;
    long lBinExcess;

    ulNrExcess = 0;  pulBinPointer = pulHistogram;
    for (i = 0; i < uiNrGreylevels; i++) { /* calculate total number of excess pixels */
        lBinExcess = (long) pulBinPointer[i] - (long) ulClipLimit;
        if (lBinExcess > 0) ulNrExcess += lBinExcess;	  /* excess in current bin */
    };

    /* Second part: clip histogram and redistribute excess pixels in each bin */
    ulBinIncr = ulNrExcess / uiNrGreylevels;		  /* average binincrement */
    ulUpper =  ulClipLimit - ulBinIncr;	 /* Bins larger than ulUpper set to cliplimit */

    for (i = 0; i < uiNrGreylevels; i++) {
        if (pulHistogram[i] > ulClipLimit) pulHistogram[i] = ulClipLimit; /* clip bin */
        else {
            if (pulHistogram[i] > ulUpper) {		/* high bin count */
                ulNrExcess -= pulHistogram[i] - ulUpper; pulHistogram[i]=ulClipLimit;
            }
            else {					/* low bin count */
                ulNrExcess -= ulBinIncr; pulHistogram[i] += ulBinIncr;
            }
        }
    }

    while (ulNrExcess) {   /* Redistribute remaining excess  */
        pulEndPointer = &pulHistogram[uiNrGreylevels]; pulHisto = pulHistogram;
        unsigned long oldExcess = ulNrExcess;
        while (ulNrExcess && pulHisto < pulEndPointer) {
            ulStepSize = uiNrGreylevels / ulNrExcess;
            if (ulStepSize < 1) ulStepSize = 1;		  /* stepsize at least 1 */
            for (pulBinPointer=pulHisto; pulBinPointer < pulEndPointer && ulNrExcess;
                pulBinPointer += ulStepSize) {
                    if (*pulBinPointer < ulClipLimit) {
                        (*pulBinPointer)++;	 ulNrExcess--;	  /* reduce excess */
                    }
            }
            pulHisto++;		  /* restart redistributing on other bin location */
        }

        if (oldExcess == ulNrExcess)
            //FIXME: failed to distribute remaining excess pixels, how to handle?
            break;
    }
}

template <typename kz_pixel_t>
void MakeHistogram (kz_pixel_t* pImage, unsigned int uiXRes,
    unsigned int uiSizeX, unsigned int uiSizeY,
    unsigned long* pulHistogram,
    unsigned int uiNrGreylevels, kz_pixel_t* pLookupTable)
    /* This function classifies the greylevels present in the array image into
    * a greylevel histogram. The pLookupTable specifies the relationship
    * between the greyvalue of the pixel (typically between 0 and 4095) and
    * the corresponding bin in the histogram (usually containing only 128 bins).
    */
{
    kz_pixel_t* pImagePointer;
    unsigned int i;

    for (i = 0; i < uiNrGreylevels; i++) pulHistogram[i] = 0L; /* clear histogram */

    for (i = 0; i < uiSizeY; i++) {
        pImagePointer = &pImage[uiSizeX];
        while (pImage < pImagePointer) pulHistogram[pLookupTable[*pImage++]]++;
        pImagePointer += uiXRes;
        pImage = &pImagePointer[-uiSizeX];
    }
}

template <typename kz_pixel_t>
void MapHistogram (unsigned long* pulHistogram, kz_pixel_t Min, kz_pixel_t Max,
    unsigned int uiNrGreylevels, unsigned long ulNrOfPixels)
    /* This function calculates the equalized lookup table (mapping) by
    * cumulating the input histogram. Note: lookup table is rescaled in range [Min..Max].
    */
{
    unsigned int i;  unsigned long ulSum = 0;
    const float fScale = ((float)(Max - Min)) / ulNrOfPixels;
    const unsigned long ulMin = (unsigned long) Min;

    for (i = 0; i < uiNrGreylevels; i++) {
        ulSum += pulHistogram[i]; pulHistogram[i]=(unsigned long)(ulMin+ulSum*fScale);
        if (pulHistogram[i] > Max) pulHistogram[i] = Max;
    }
}

template <typename kz_pixel_t>
void MakeLut (kz_pixel_t * pLUT, kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrBins)
    /* To speed up histogram clipping, the input image [Min,Max] is scaled down to
    * [0,uiNrBins-1]. This function calculates the LUT.
    */
{
    int i;
    const kz_pixel_t BinSize = (kz_pixel_t) (1 + (Max - Min) / uiNrBins);

    for (i = Min; i <= Max; i++)  pLUT[i] = (i - Min) / BinSize;
}

template <typename kz_pixel_t>
void Interpolate (kz_pixel_t * pImage, int uiXRes, unsigned long * pulMapLU,
    unsigned long * pulMapRU, unsigned long * pulMapLB,  unsigned long * pulMapRB,
    unsigned int uiXSize, unsigned int uiYSize, kz_pixel_t * pLUT)
    /* pImage      - pointer to input/output image
    * uiXRes      - resolution of image in x-direction
    * pulMap*     - mappings of greylevels from histograms
    * uiXSize     - uiXSize of image submatrix
    * uiYSize     - uiYSize of image submatrix
    * pLUT	       - lookup table containing mapping greyvalues to bins
    * This function calculates the new greylevel assignments of pixels within a submatrix
    * of the image with size uiXSize and uiYSize. This is done by a bilinear interpolation
    * between four different mappings in order to eliminate boundary artifacts.
    * It uses a division; since division is often an expensive operation, I added code to
    * perform a logical shift instead when feasible.
    */
{
    const unsigned int uiIncr = uiXRes-uiXSize; /* Pointer increment after processing row */
    kz_pixel_t GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;

    if (uiNum & (uiNum - 1))   /* If uiNum is not a power of two, use division */
        for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;
            uiYCoef++, uiYInvCoef--,pImage+=uiIncr) {
                for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;
                    uiXCoef++, uiXInvCoef--) {
                        GreyValue = pLUT[*pImage];		   /* get histogram bin value */
                        *pImage++ = (kz_pixel_t ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue]
                        + uiXCoef * pulMapRU[GreyValue])
                            + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue]
                        + uiXCoef * pulMapRB[GreyValue])) / uiNum);
                }
        }
    else {			   /* avoid the division and use a right shift instead */
        while (uiNum >>= 1) uiShift++;		   /* Calculate 2log of uiNum */
        for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;
            uiYCoef++, uiYInvCoef--,pImage+=uiIncr) {
                for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;
                    uiXCoef++, uiXInvCoef--) {
                        GreyValue = pLUT[*pImage];	  /* get histogram bin value */
                        *pImage++ = (kz_pixel_t)((uiYInvCoef* (uiXInvCoef * pulMapLU[GreyValue]
                        + uiXCoef * pulMapRU[GreyValue])
                            + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue]
                        + uiXCoef * pulMapRB[GreyValue])) >> uiShift);
                }
        }
    }
}
