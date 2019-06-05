import numpy as np
import os,sys
import gdal
from tqdm import tqdm
import argparse
from base_functions import get_file

parser=argparse.ArgumentParser(description='images stretching for normalization')
parser.add_argument('--input', dest='input_dir', help='images input directory',
                         default='D:\\data\\test\\images_original\\')
parser.add_argument('--output', dest='output_dir', help='new images saving directory',
                         default='D:\\data\\test\\normal\\')
parser.add_argument('--dtype', dest='dtype', help='8bits or 16bits',
                         default='16bits')
args=parser.parse_args()

if __name__=="__main__":
    input_dir = args.input_dir
    output_dir = args.output_dir
    dtype = args.dtype

    if not os.path.isdir(input_dir):
        print("Error: output dir do not exist!")
        sys.exit(-1)

    if not os.path.isdir(output_dir):
        print("Warning: output dir do not exist!")
        os.mkdir(output_dir)

    nodata = 65535.0
    result_bits = '16bits'
    valid_range = 1024
    cut_value = 100

    if '8' in dtype:
        nodata = 255
        result_bits = '8bits'
        valid_range = 255
        cut_value = 25
    else:
        pass

    src_files, tt = get_file(input_dir)
    assert (tt != 0)
    factor = 4.0

    if '8' in result_bits:
        assert (valid_range < 256)
        factor = 6.0
    elif '16' in result_bits:
        assert (valid_range < 65536)
        factor = 4.0
    else:
        pass

    for file in tqdm(src_files):

        absname = os.path.split(file)[1]
        # absname = absname.split('.')[0]
        # absname = ''.join([absname, '.tif'])
        print(absname)
        if not os.path.isfile(file):
            print("input file dose not exist:{}\n".format(file))
            # sys.exit(-1)
            continue

        dataset = gdal.Open(file)
        if dataset == None:
            print("Open file failed: {}".format(file))
            continue

        height = dataset.RasterYSize
        width = dataset.RasterXSize
        im_bands = dataset.RasterCount
        assert(im_bands>1)
        im_type = dataset.GetRasterBand(1).DataType
        img = dataset.ReadAsArray(0, 0, width, height)
        geotransform = dataset.GetGeoTransform()
        del dataset

        img = np.array(img, np.float32)
        result = []
        for i in range(im_bands):
            data = np.array(img[i])
            maxium = data.max()
            minm = data.min()
            mean = data.mean()
            std = data.std()
            print("\nOriginal max, min, mean,std:[{},{},{},{}]".format(maxium, minm, mean, std))
            data = data.reshape(height * width)
            ind = np.where((data > 0) & (data < nodata))
            ind = np.array(ind)

            a, b = ind.shape
            print("valid value number: {}".format(b))
            # tmp = np.zeros(b, np.uint16)
            tmp = np.zeros(b, np.float32)
            for j in range(b):
                tmp[j] = data[ind[0, j]]
            tmaxium = tmp.max()
            tminm = tmp.min()
            tmean = tmp.mean()
            tstd = tmp.std()
            # print(tmaxium, tminm, tmean, tstd)
            tt = (data - tmean) / tstd  # first Z-score normalization
            tt = (tt + factor) * valid_range / (2 * factor) - cut_value
            tind = np.where(data == 0)

            tt = np.array(tt)
            # tt = tt.astype(np.uint8)
            tt = tt.astype(np.uint16)
            tt[tind] = 0

            smaxium = tt.max()
            sminm = tt.min()
            smean = tt.mean()
            sstd = tt.std()
            # print(smaxium, sminm, smean, sstd)
            print("New max, min, mean,std:[{},{},{},{}]".format(smaxium, sminm, smean, sstd))

            out = tt.reshape((height, width))
            result.append(out)

        outputfile = os.path.join(output_dir, absname)
        driver = gdal.GetDriverByName("GTiff")

        if '8' in result_bits:
            outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_Byte)
            outdataset.SetGeoTransform(geotransform)
        elif '16' in result_bits:
            outdataset = driver.Create(outputfile, width, height, im_bands, gdal.GDT_UInt16)
            outdataset.SetGeoTransform(geotransform)

        for i in range(im_bands):
            outdataset.GetRasterBand(i + 1).WriteArray(result[i])

        del outdataset
