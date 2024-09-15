import dateutil.tz
from datetime import datetime

# 训练模型输出路径
def generate_dir2save(opt):
    # training_image_name=berea
    training_image_name = opt.input_name.split("/")[-1]
    dir2save = '/home1/Usr/yinmengkai/SelfStudy/GNGAN3D/outf/TrainedModels/{}/'.format(training_image_name)
    # 获取当前时间
    timestamp = datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    dir2save += timestamp
    return dir2save