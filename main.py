from func.ffmpeg.cutScore import score2part, singleScore
from func.ffmpeg.predict import batch_predict, single_predict
from func.ffmpeg.txt2music import note2chime, note2music


# 作用：将五线谱图像转变为音乐
# img表示图片路径, 不可含有中文
# isChime表示是否生成编钟音乐，True表示编钟，False表示钢琴
# output_name表示生成音乐的名称，不要加后缀，最后会默认生成音乐在output文件夹中
def imgs2music(img_name, output_name='1'):
    imgs = score2part(img_name)
    notestxt = batch_predict(imgs)
    if len(notestxt)==0:
        return 0
    else:
        note2chime(notestxt, output_name)
        note2music(notestxt, output_name)
        return 1


# 作用：将单行五线谱图像转变为音乐
# img表示图片路径，不可含有中文
# isChime表示是否生成编钟音乐，True表示编钟，False表示钢琴
# output_name表示生成音乐的名称，不要加后缀，最后会默认生成音乐在output文件夹中
def singleImg2music(img_name, isChime, output_name):
    img = singleScore(img_name)
    notestxt = single_predict(img)
    note2chime(notestxt, output_name)
    note2music(notestxt, output_name)


if __name__ == '__main__':
    filename = 'testImages/HappyBirthday.png'
    imgs2music(filename, '1')
