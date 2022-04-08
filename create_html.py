import os
# import json
import yaml
from torch.cuda import get_device_properties, device_count


def get_gpu_number():
    total = device_count()
    gpu_list = []
    for i in range(total):
        msg = get_device_properties(i)
        gpu_list.append("%s（%dMB）" % (msg.name, int(msg.total_memory/1024**2)))
    #import os
    # total = 0
    #gpu_list = os.popen("nvidia-smi -L").readlines()
    #for gpu_msg in gpu_list:
    #    if gpu_msg.startswith('GPU'):
    #        total += 1
    return total, gpu_list


def get_dirs_and_annos(data_root):
    from glob import glob
    import os
    all_list = []
    # print("%s*" % data_root, glob("%s*" % data_root), os.getcwd())
    for i, name in enumerate(glob("%s*" % data_root)):
        name = name.replace('\\', '/')
        # print(name)
        if os.path.isdir(name) and not name.endswith("/annotations"):
            name = name[len("%s" % data_root):]
            all_list.append(name)
    # print('all list', all_list)
    # 子目录annotations中所有json文件
    anno_list = []
    for json_file in glob("%sannotations/*.json" % data_root):
        json_file = json_file.replace('\\', '/')
        # print(json_file)
        if os.path.isfile(json_file):
            anno_list.append(json_file[len("%sannotations/" % data_root):])
    return sorted(all_list), sorted(anno_list)


def is_training(setting_name):
    is_training = False
    if os.path.exists('./settings/%s/output/pid' % setting_name):
        pids = open('./settings/%s/output/pid' % setting_name).read().split('\n')
        for this_pid in pids:
            if len(this_pid):
                if os.path.isdir('/proc/%s' % this_pid):
                    is_training = True
                    break
    return is_training


def stop_all_pid(setting_name):
    if os.path.exists('./settings/%s/output/pid' % setting_name):
        pids = open('./settings/%s/output/pid' % setting_name).read().split('\n')
        for this_pid in pids:
            if len(this_pid):
                if os.path.isdir('/proc/%s' % this_pid):
                    os.popen('kill -9 %s' % this_pid)
        os.remove('./settings/%s/output/pid' % setting_name)


def get_file_size(file_name, unit='M'):
    units = {
        "B": 0,
        "K": 1,
        "M": 2,
        "G": 3,
        "T": 4
    }
    return os.path.getsize(file_name) / float(1024 ** units[unit])


#########################################################################################
def extract_train_msg(setting_dir):
    setting_dir = './settings/%s' % setting_dir
    now_epoch = int(open(os.path.join(setting_dir, 'output', 'now_epoch')).read())
    now_epoch_dir = os.path.join(setting_dir, 'output', 'epochs', str(now_epoch).zfill(4))
    # print(setting_dir)
    # print(now_epoch_dir)
    total = open(os.path.join(now_epoch_dir, 'total')).read().split('\n')
    once = ["now_epoch", "total_epoch", "total_iter", "memory_use"]
    others = ["now_iter", "lr", "total_loss", "iou_loss", "l1_loss", "conf_loss", "cls_loss"]
    msg_send = {}
    now_msg = {}
    now_iter = 0
    for name in others:
        msg_send[name] = []
    for line in total:
        if len(line):
            now_msg = yaml.load(line, yaml.Loader)
            if now_iter < now_msg["now_iter"]:
                now_iter = now_msg["now_iter"]
                for name in others:
                    msg_send[name].append(now_msg[name])
    for name in once:
        msg_send[name] = now_msg[name]

    msg_send['epoch_total_loss'] = [0]
    msg_send['epoch_iou_loss'] = [0]
    msg_send['epoch_l1_loss'] = [0]
    msg_send['epoch_conf_loss'] = [0]
    msg_send['epoch_cls_loss'] = [0]
    msg_send['epoch_pre'] = [0]
    for previews_epoch in range(now_epoch):
        previews_epoch_dir = os.path.join(setting_dir, 'output', 'epochs', str(previews_epoch + 1).zfill(4))
        if os.path.isdir(previews_epoch_dir):

            this_pre_data_file = os.path.join(previews_epoch_dir, 'latest')
            if os.path.isfile(os.path.join(previews_epoch_dir, 'final')):
                this_pre_data_file = os.path.join(previews_epoch_dir, 'final')

            this_pre_data = yaml.load(open(this_pre_data_file).read(), yaml.Loader)

            msg_send['epoch_total_loss'].append(this_pre_data["total_loss"])
            msg_send['epoch_iou_loss'].append(this_pre_data["iou_loss"])
            msg_send['epoch_l1_loss'].append(this_pre_data["l1_loss"])
            msg_send['epoch_conf_loss'].append(this_pre_data["conf_loss"])
            msg_send['epoch_cls_loss'].append(this_pre_data["cls_loss"])
            msg_send['epoch_pre'].append(previews_epoch + 1)

    return msg_send


def extract_latest_train_msg(setting_dir, epoch: int or None = None):
    setting_dir = './settings/%s' % setting_dir
    now_epoch = int(open(os.path.join(setting_dir, 'output', 'now_epoch')).read()) if epoch is None else epoch
    now_epoch_dir = os.path.join(setting_dir, 'output', 'epochs', str(now_epoch).zfill(4))
    return open(os.path.join(now_epoch_dir, 'latest')).read()


def extract_eval_msg(setting_dir):
    eval_msg = {
        "ap50_95": [0.0],
        "ap50": [0.0],
        "now_epoch": [0],
    }

    if os.path.isfile('./settings/%s/output/eval' % setting_dir):
        total = open('./settings/%s/output/eval' % setting_dir).read().split('\n')
        for line in total:
            if len(line):
                now_msg = yaml.load(line, yaml.Loader)
                if eval_msg["now_epoch"][-1] == now_msg["now_epoch"]:
                    eval_msg["ap50_95"][-1] = max(eval_msg["ap50_95"][-1], now_msg["ap50_95"])
                    eval_msg["ap50"][-1] = max(eval_msg["ap50"][-1], now_msg["ap50"])
                else:
                    eval_msg["ap50_95"].append(now_msg["ap50_95"])
                    eval_msg["ap50"].append(now_msg["ap50"])
                    eval_msg["now_epoch"].append(now_msg["now_epoch"])
    return eval_msg


def get_all_pth_files_list(setting_dir):
    from glob import glob
    table_type = ["序号", "文件名称", "大小", "操作"]
    pth_file_dir = './settings/%s/output/' % setting_dir
    string_head = ""
    string_show = ""
    for this_type in table_type:
        string_head += "<th>%s</th>" % this_type
    for i, pth_file_name in enumerate(sorted(glob(pth_file_dir + '*.pth'))):
        file_size = get_file_size(pth_file_name)
        pth_file_name = pth_file_name.split('/')[-1]
        string_show += """        <tr>
            <td><center>%d</center></td>
            <td><center>%s</center></td>
            <td><center>%.2f MB</center></td>
            <td><center>%s</center></td>
        </tr>""" % (
            i + 1,
            pth_file_name,
            file_size,
            '<a href="/pth/%s?name=%s">下载</a>&nbsp;&nbsp;'
            '<a href=javascript:send_request("/delete_pth?from=%s&name=%s","确定要删除%s吗？")>删除</a>' % (
                pth_file_name,
                setting_dir,
                setting_dir,
                pth_file_name,
                pth_file_name
            )
        )
    return "<table style='width: 100%%'>%s%s</table>" % (string_head, string_show if len(string_show) else '')


#########################################################################################
# 以下是完整的html
def jump2Html(url, text: list or None = None, time_delay=0.):
    text = [] if text is None else [text] if isinstance(text, str) else text
    assert isinstance(text, list)
    msg = ""
    for line in text:
        msg += "<p>%s</p>" % line
    html_string = """<!DOCTYPE html>
<html>
    <head>
        <title>WEB-YOLOX 跳转提示</title>
        <meta http-equiv="refresh" content="%s;url=%s">
    </head>
    <body>
        %s
    </body>
</html>""" % (time_delay, url, msg)
    return html_string


def getBasicSettingsHtml(data_root, settings_dir):
    import os
    import yaml
    all_model_size = {
        's': 'small',
        'm': 'medium',
        'l': 'large',
        'x': 'extra large',
        'tiny': 'tiny'
    }
    gpu_num, gpu_msg = get_gpu_number()
    data_root = os.path.abspath(data_root).replace('\\', '/')
    if not data_root.endswith('/'):
        data_root += '/'
    open('./settings/%s/data_dir.txt' % settings_dir, 'w').write(data_root)
    dir_list, anno_list = get_dirs_and_annos(data_root)

    msg = {}
    settings_exist = False
    if os.path.isfile('./settings/%s/settings.yaml' % settings_dir):
        settings_exist = True
        msg = yaml.load(open('./settings/%s/settings.yaml' % settings_dir), yaml.Loader)
    print(msg)
    train_dir_index = dir_list.index(msg['train_dataset_path'].split('/')[-1]) if settings_exist else 0
    val_dir_index = dir_list.index(msg['val_dataset_path'].split('/')[-1]) if settings_exist else 0
    train_json_index = anno_list.index(msg['train_annotation_file'].split('/')[-1]) if settings_exist else 0
    val_json_index = anno_list.index(msg['val_annotation_file'].split('/')[-1]) if settings_exist else 0
    model_size = msg['model_size'] if settings_exist else 's'
    use_gpu_num = msg['gpu_num'] if settings_exist else 0
    epochs = msg['epochs'] if settings_exist else 100
    batch_size = msg['batch_size'] if settings_exist else 12
    start_epoch = msg['start_epoch'] if settings_exist else 0
    pretrained_file = '' if not settings_exist else '' if msg['pretrained_weight_file'] == 'no file selected' else msg['pretrained_weight_file']
    fp16 = msg['fp16'] if settings_exist else False
    save_each_epoch = msg['save_each_epoch'] if settings_exist else False

    def gpu_choose2str(my_list):
        this_string = ""
        for item in my_list:
            this_string += "%s;" % str(item + 1)
        return this_string[:-1]

    gpu_choose = gpu_choose2str(msg["gpu_choose"]) if "gpu_choose" in msg else ""

    gpu_msg_string = "可用GPU:<br />"
    train_dir_string = ""
    val_dir_string = ""
    train_anno_string = ""
    val_anno_string = ""
    model_size_string = ""
    gpu_string = ""


    for now_dir in dir_list:
        train_dir_string += "<option value='%s'%s>%s</option>" % (
            now_dir,
            ' selected'
            if now_dir == dir_list[train_dir_index]
            else '',
            now_dir
        )
        val_dir_string += "<option value='%s'%s>%s</option>" % (
            now_dir,
            ' selected'
            if now_dir == dir_list[val_dir_index]
            else '',
            now_dir
        )

    for now_anno in anno_list:
        train_anno_string += "<option value='%s'%s>%s</option>" % (
            now_anno,
            ' selected'
            if now_anno == anno_list[train_json_index]
            else '',
            now_anno
        )
        val_anno_string += "<option value='%s'%s>%s</option>" % (
            now_anno,
            ' selected'
            if now_anno == anno_list[val_json_index]
            else '',
            now_anno
        )

    for this_model_size in all_model_size:
        model_size_string += '<option value="%s"%s>%s(%s)</option>' % (
            this_model_size,
            ' selected'
            if this_model_size == model_size
            else '',
            all_model_size[this_model_size],
            this_model_size
        )

    for i in range(gpu_num):
        gpu_string += "<option value=%d%s>%d</option>" % (
            i + 1,
            ' selected'
            if i + 1 == use_gpu_num
            else '',
            i + 1
        )
        gpu_msg_string += "%d. <strong style='color:rgb(65,250,80);'>%s</strong><br />" % (
            i + 1,
            gpu_msg[i].split('(')[0].split(':')[-1]
        )


    strings = (
        gpu_msg_string,
        settings_dir,
        data_root,
        train_dir_string,
        val_dir_string,
        train_anno_string,
        val_anno_string,
        model_size_string,
        gpu_string,
        epochs,
        batch_size,
        start_epoch,
        settings_dir,
        pretrained_file,
        gpu_choose,
        ' checked' if fp16 else '',
        ' checked' if save_each_epoch else ''
    )
    return """<html>
    <head>
        <title>WEB-YOLOX 训练基本配置</title>
        <meta charset="utf-8">
        <link rel="stylesheet" type="text/css" href="file/settings.css?new=true">
    </head>
    <body>
        <form
            action="/save_basic_settings"
            method="post"
            enctype="multiple/form-data"
            class="elegant-aero"
        >
        <h3>YOLOX 训练基本配置</h3>
        %s<br />
        <div>
            训练名称:<br />
            <input name="train_name" type="text" value="%s" readonly required><br />
        </div>
        <div>
            数据集根目录:<br />
            <input name="data_dir" type="text" value="%s" readonly required><br />
        </div>
        <div class="selectline">
            <div class="select">
                训练集目录:<br />
                <select name="train_dir" required>%s</select><br />
            </div>
            <div class="select">
                验证集目录:<br />
                <select name="val_dir" required>%s</select><br />
            </div>
        </div>
        <div class="selectline">
            <div class="select">
                训练集标签文件:<br />
                <select name="train_anno" required>%s</select><br />
            </div>
            <div class="select">
                验证集标签文件:<br />
                <select name = "val_anno" required>%s</select><br />
            </div>
        </div>
        <div class="selectline">
            <div class="select">
                模型大小:<br />
                <select name = "model_size" required>
                    %s
                </select><br />
            </div>
            <div class="select">
                使用gpu个数:<br />
                <select name = "gpu_num" required>%s</select><br />
            </div>
        </div>
        <div class="numberline">
            <div class="number">
                训练次数(total epochs):<br />
                <input type="number" name="epoch" value="%s" required/><br />
            </div>
            <div class="number">
                批次大小(batch size):<br />
                <input type="number" name="batch_size" value="%s" required/><br />
            </div>
            <div class="number">
                已训练次数（中断后可继续训练）:<br />
                <input type="number" name="start_epoch" value="%s" required/><br />
            </div>
        </div>
        <div>
            预训练模型文件:<br />
            <input name="pretrained_weight_file" placeholder="不使用不填，大小一定要对应(backbone file:./weight/XX_backbone.pth, ckpt file:./settings/%s/output/last.pth)" type="text" value="%s"><br />
        </div>
        <div>
            使用的GPU编号:<br />
            <input type="text" placeholder="参考可用GPU编号，用分号隔开，不填则按编号顺序使用GPU（示例：1;3）" name="gpu_choose" value="%s"/><br />
        </div>
            <input type="checkbox"%s name="fp16">使用混合精度训练(fp16)<br />
            <input type="checkbox"%s name="save_each_epoch">每次训练均保存模型<br />
            
            <br /><textarea name="classes" placeholder="即将推出"></textarea><br />
        
            <br />
            <input type="reset" class="button" value="重置"> &nbsp;
            <input type="submit" class="button" value="提交"><br />
            <input type="hidden" name="token" value="null">

        </form>
    </body>
</html>
""" % strings


def getHypSettingsHtml(settings_dir):

    if os.path.isdir('./settings/%s' % settings_dir):
        hyp_file_name = './settings/%s/hyp.yaml' % settings_dir
        assert os.path.isfile(hyp_file_name)
        msg = yaml.load(open(hyp_file_name), yaml.Loader)
        warmup_epochs = msg["warmup_epochs"]
        warmup_lr = msg["warmup_lr"]
        basic_lr_per_img = msg["basic_lr_per_img"]
        no_aug_epochs = msg["no_aug_epochs"]
        min_lr_ratio = msg["min_lr_ratio"]
        ema = msg["ema"]
        weight_decay = msg["weight_decay"]
        momentum = msg["momentum"]
        print_interval = msg["print_interval"]
        eval_interval = msg["eval_interval"]

        strings = (
            settings_dir,
            str(warmup_epochs),
            str(weight_decay),
            str(momentum),
            str(warmup_lr),
            str(basic_lr_per_img),
            str(min_lr_ratio),
            str(no_aug_epochs),
            str(print_interval),
            str(eval_interval),
            ' checked' if ema else ''
        )

        return """<html>
    <head>
        <title>WEB-YOLOX 训练超参数配置</title>
        <meta charset="utf-8">
        <link rel="stylesheet" type="text/css" href="file/hyps.css?new=true">
    </head>
    <body>
        <form
            action="/save_hyp"
            method="post"
            enctype="multiple/form-data"
            class="elegant-aero"
        >
        <h3>YOLOX训练超参数配置</h3><br /><br />
            正在配置: <input type="text" name="name" value="%s" readonly required/><br /><br />
            <div class="numberline">
                <div class="number">
                    学习率预热次数<br />(warmup epochs):
                    <input type="number" name="warmup_epochs" value="%s" required/>
                </div>
                <div class="number">
                    权重衰减系数<br />(weight decay):
                    <input type="number" name="weight_decay" value="%s" step="any" required/>
                </div>
                <div class="number">
                    动量<br />(momentum):
                    <input type="number" name="momentum" value="%s" step="any" required/>
                </div>
            </div>
            <div class="numberline">
                <div class="number">
                    预热起始学习率<br />(warmup learning rate):
                    <input type="number" name="warmup_lr" value="%s" step="any" required/>
                </div>
                <div class="number">
                    每张图片的初始学习率<br />(basic learning rate):
                    <input type="number" name="basic_lr_per_img" value="%s" step="any" required/>
                </div>
                <div class="number">
                    最终学习率的比值<br />(final lr / basic lr):
                    <input type="number" name="min_lr_ratio" value="%s" step="any" required/>
                </div>
            </div>
            <div class="numberline">
                <div class="number">
                    末尾停用数据增强次数<br />(no aug epochs):
                    <input type="number" name="no_aug_epochs" value="%s" required/>
                </div>
                <div class="number">
                    输出训练结果间隔<br />(print per iteration)
                    <input type="number" name="print_interval" value="%s" required/>
                </div>
                <div class="number">
                    验证间隔<br />(validate per epoch)<br />
                    <input type="number" name="eval_interval" value="%s" required/><br />
                </div>
            </div>

            <br />
            <input type="checkbox" name="ema"%s />使用EMA策略(提升模型鲁棒性和性能)<br />


            <br /><br />
            <input type="reset" class="button" value="重置"> &nbsp;
            <input type="submit" class="button" value="提交"><br />
        </form>
    </body>
</html>
""" % strings
    else:
        return jump2Html('/settings_list', 'Invalid url!', 1)


def getSettingsListHtml():
    from glob import glob
    list_show = []
    table_type = ["序号", "训练配置名称", "配置完成情况", "训练情况", "操作"]
    now_interpreter = open('./run/interpreter.txt').read()
    string_show = ""
    for this_type in table_type:
        string_show += "<th>%s</th>" % this_type
    string_show = "<tr>%s</tr>" % string_show
    for path in sorted(glob('./settings/*')):
        # print(path)
        if os.path.isdir(path):
            list_show.append({
                'name': path.split('/')[-1],
                'ok': os.path.isfile(os.path.join(path, 'settings.yaml'))
            })

    for i, item in enumerate(list_show):
        is_train = is_training(item['name'])
        train_command = 'start_train' if not is_train else 'stop_train'
        train_cmd_show = '开始训练' if not is_train else '终止训练'
        train_show = '<a href="/%s?name=%s">%s</a>&nbsp;&nbsp;' % (train_command, item['name'], train_cmd_show) \
                     if item['ok'] else ''
        string_show += """<tr>
        <td><center>%d</center></td>
        <td><center>%s</center></td>
        <td><center>%s</center></td>
        <td><center>%s</center></td>
        <td><center>%s</center></td>
        </tr>""" % (
            i + 1,
            '%s&nbsp;&nbsp;<a href="/train_details?name=%s&show_length=30" target="_blank">详情</a>' % (item['name'], item['name']),
            '<text style="color:green">已完成配置</text>' if item['ok'] else '<text style="color:red">未完成配置</text>',
            '<text style="color:green">正在训练, <a href="/train_log?name=%s&line_length=30" style="color:green" target="_blank">点此查看日志</a></text>' % item['name'] if is_train else '<text style="color:red">未在训练</text>',
            '<a href="/edit_setting?name=%s">修改基本配置</a>&nbsp;&nbsp;'
            '<a href="/edit_hyp?name=%s">修改超参数</a>&nbsp;&nbsp;'
            '%s'
            '%s' % (
                item['name'],
                item['name'],
                train_show,
                ('<a href="/delete_setting?name=%s">删除该配置</a>' % item['name']) if not is_train else ''
            )
        )

    string_show = """<html> 
    <head> 
        <title>WEB-YOLOX 训练配置列表</title> 
        <link rel="stylesheet" type="text/css" href="file/table.css?new=true"> 
    </head> 
    <body> 
        <div style="display: flex;"> 
            <div style="width: 50%%"><a href="/set_interpreter">设置解释器</a>&nbsp;&nbsp;当前解释器: %s</div> 
            <div style="width: 50%%"> 
                <div style="float: right;padding-right: 10px;"> 
                    <a href="/file/yolox_base.zip">下载测试代码</a>&nbsp;&nbsp; 
                    <a href="/change_user_pwd">修改用户名和密码</a>&nbsp;&nbsp; 
                    <a href="/logout">注销</a> 
                </div> 
            </div> 
        </div> 
        <center> 
            <h2>训练配置列表</h2> 
            <a href="/create_new_setting">创建新的训练配置</a> 
            <table>%s</table> 
        </center> 
    </body> 
</html>""" % (now_interpreter, string_show)
    return string_show


def getSetInterpreterHtml():
    now_interpreter = open('./run/interpreter.txt').read()
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WEB-YOLOX 选择解释器</title>
    <link rel="stylesheet" type="text/css" href="file/login.css"/>
</head>
<body>
    <div id="create">
        <center><h1>选择解释器</h1></center>
        <form method="post" action="">
            <p><center>
            <input type="text" placeholder="例：/root/anaconda3/envs/yolo/bin/python3" name="interpreter" value="%s" required />
            </center></p>
            <center><button class="but" type="submit">确认更改</button></center><br>
            <center><button class="but" type="reset">重置</button></center>
        </form>
    </div>
</body>
</html>""" % now_interpreter


def confirmDeleteSettingHtml(name: str):
    string_show = """<!DOCTYPE html> 
<html lang="en"> 
<html> 
    <head> 
        <meta charset="UTF-8"> 
        <title>WEB-YOLOX 确认删除配置</title> 
        <link rel="stylesheet" type="text/css" href="file/login.css?new=true"> 
    </head> 
    <body> 
        <center> 
            <div id="create"> 
                <h1>您正要删除配置: %s</h1> 
                <form id="confirm" name="confirm" action="/confirm_delete" method="post"> 
                     <div id="msg" style="color:red"></div> 
                     <p><input type="text" name="inputname" placeholder="请输入该配置名称以确认删除" required/></p> 
                     <input type="hidden" name="name" value="%s" required/> 
                     <button class="but" type="button" onclick="confirm_and_submit();">确认删除</button> 
                </form> 
            </div> 
        </center> 
        <script type="text/javascript"> 
            confirm_and_submit = function () { 
                var confirm = document.getElementById("confirm"); 
                var real_name = confirm.name.value; 
                var input_name = confirm.inputname.value; 
                var msg = document.getElementById("msg"); 
                if (real_name === input_name) { 
                    confirm.submit(); 
                } 
                else { 
                    msg.innerHTML = "<p>输入错误！</p>"; 
                    console.log("wrong input: " + msg.value); 
                } 
            } 
        </script> 
        <!-script src="/file/confirm_delete.js?new=true" /> 
    </body> 
</html>""" % (name, name)
    return string_show






