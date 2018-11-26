from tongue_analysis import *

def info_save(result):
    # dict(result) {'mosaic_img':打码图片, 'tongue_img':舌头图片, 'shezhi':舌质结果, 'shetai':舌苔结果}
    mosaic = result['mosaic_img']
    tongue = result['tongue_img']
    shezhi = result['shezhi']
    shetai = result['shetai']

    mosaic = Image.fromarray(mosaic)
    mosaic.save('mosaic.jpg','jpeg')
    tongue = Image.fromarray(tongue)
    tongue.save('tongue.jpg','jpeg')


if __name__ == '__main__':
    # init
    sess = setup()
    # print('start')
    # time.sleep(10)
    # 内存消耗 822M

    # while working:
    t = time.time()
    result = analysis('test.jpg', sess=sess)
    info_save(result)
    print(time.time()-t)
    # 时间4.2秒，内存消耗峰值2082M，其中舌象定位3.7秒
    time.sleep(5)

    t = time.time()
    result = analysis('test1.jpg', sess=sess)
    info_save(result)
    print(time.time()-t)
    # 时间0.42秒

    t = time.time()
    result = analysis('test.jpg', sess=sess)
    info_save(result)
    print(time.time()-t)
    # 时间0.42秒


    # exit
    time.sleep(10)
    sess.close()
