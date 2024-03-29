"""
author: LSH9832
"""
import os
import argparse
from datetime import timedelta
from glob import glob
from flask import *
import yaml

import platform

from screen import Screen
import create_html


this_file_path = os.path.abspath(__file__)
os.chdir(os.path.abspath(os.path.dirname(this_file_path)))
# print(os.getcwd())

app = Flask(create_html.get_web_name(), template_folder="./html")
app.config['SECRET_KEY'] = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=7)      # 7天内免登录
SYSTEM = platform.system()


#############################################################################
def make_parser():
    parser = argparse.ArgumentParser("%s parser" % create_html.get_web_name())
    parser.add_argument("-p", "--port", type=int, default=8080, help="running port")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    parser.add_argument("--log-off", default=False, action="store_true", help="turn off logger")
    return parser


#############################################################################
# 定义用户名和密码
def get_user_pwd():
    return yaml.load(open('./run/user.yaml'), yaml.Loader)


# 获取网页传回的表单
def get_webform():
    return request.form.to_dict()


# 获取网页的get请求
def get_web_get():
    return request.args.to_dict()


# 判断是否为移动端访问
def is_mobile_device():
    flag = not ('Windows' in str(request.user_agent))
    return flag


def login_already():
    return session.get('user') in get_user_pwd()


#############################################################################
def login_base():
    if login_already():
        return create_html.jump2Html('/')
    msg = get_webform()
    if 'username' in msg and 'password' in msg:
        if {msg['username']: msg['password']} == get_user_pwd():
            session['user'] = msg['username']
            return create_html.jump2Html('/', '登录成功', 1)
        return create_html.jump2Html('/login', '登录失败！', 1)
    return render_template('login.html', web_name=create_html.get_web_name())


def logout_base():
    while session['user'] is not None:
        session['user'] = None
    return create_html.jump2Html('/login', '注销成功', 1)


def index():
    return create_html.jump2Html('/settings_list')


#############################################################################
@app.before_request
def check_login():
    if request.path.startswith('/file') or request.path.startswith('/login'):
        return
    else:
        if session.get('user') in get_user_pwd():
            return
        return create_html.jump2Html('/login')


#############################################################################
# 相关的文件
@app.route('/file/<path>')
def file(path: str):
    # checkDevice()
    file_type = path.split('.')[-1]
    file_path = './file/%s/%s' % (file_type, path)
    print(file_path)
    return send_file(file_path)


@app.route('/pth/<path>', methods=["GET", "POST"])
def pth_file(path: str):
    # checkDevice()
    msg = get_web_get()
    if 'name' in msg:
        file_type = path.split('.')[-1]
        if file_type == 'pth':
            file_path = './settings/%s/output/%s' % (msg['name'], path)
            print(file_path)
            return send_file(file_path)
        return create_html.jump2Html('/train_details?name=%s&show_length=30' % msg['name'], '只能下载权重文件哦', 1)
    return create_html.jump2Html('/settings_list', '小老弟想啥呢', 1)


#############################################################################
# 解释器配置
@app.route('/set_interpreter', methods=["GET", "POST"])
def set_interpreter():
    msg = get_webform()
    if 'interpreter' in msg:
        open('./run/interpreter.txt', 'w').write(msg['interpreter'])
        return create_html.jump2Html('/settings_list', 'success', 1)
    return create_html.getSetInterpreterHtml()


# 修改用户名和密码
@app.route('/change_user_pwd', methods=["GET", "POST"])
def change_user_pwd():
    msg = get_webform()
    if 'new_username' in msg and 'new_password' in msg:
        yaml.dump({msg['new_username']: msg['new_password']}, open('./run/user.yaml', 'w'), yaml.Dumper)
        return create_html.jump2Html('/settings_list', 'success', 1)
    return render_template('change_user_pwd.html', web_name=create_html.get_web_name())


#############################################################################
# 参数配置 增删改查
@app.route('/create_new_setting', methods=["GET", "POST"])
def create_setting():
    return render_template('yolox_create.html', web_name=create_html.get_web_name())


@app.route('/basic_settings', methods=["GET", "POST"])
def basic_settings():
    msg = get_webform()
    print(msg)
    assert 'data_dir' in msg and 'train_name' in msg
    assert ('/' not in msg['train_name']) and ('*' not in msg['train_name'])
    new_dir_name = './settings/%s' % msg['train_name']
    if not os.path.isdir(new_dir_name):
        os.makedirs(new_dir_name)
    hyp_file = '%s/hyp.yaml' % new_dir_name
    if not os.path.exists(hyp_file):
        open(hyp_file, 'w').write(open('./hyps/default.yaml').read())
    # print(os.getcwd())
    return create_html.getBasicSettingsHtml(msg['data_dir'], msg['train_name'])


@app.route('/edit_setting', methods=["GET", "POST"])
def edit_setting():
    msg = get_web_get()
    # print(msg)
    assert 'name' in msg
    if os.path.isfile('./settings/%s/data_dir.txt' % msg['name']):
        data_dir = open('./settings/%s/data_dir.txt' % msg['name']).read()
        return create_html.getBasicSettingsHtml(data_dir, msg['name'])
    else:
        return create_html.jump2Html('/settings_list', 'error', 1)


@app.route('/edit_hyp', methods=["GET", "POST"])
def edit_hyp():
    msg = get_web_get()
    print(msg)
    assert 'name' in msg
    if os.path.isdir('./settings/%s' % msg["name"]):
        return create_html.getHypSettingsHtml(msg["name"])
    else:
        return create_html.jump2Html('/settings_list', 'error', 1)


@app.route('/save_basic_settings', methods=["GET", "POST"])
def save_basic():
    msg = get_webform()
    for name in msg:
        print(name, msg[name])
    output_dir = os.path.abspath("./settings/%s/" % msg['train_name']).replace('\\', '/')
    use_pretrained_file = bool(len(msg['pretrained_weight_file']))
    pretrained_file = os.path.abspath(msg['pretrained_weight_file']).replace('\\', '/')
    if not output_dir.endswith('/'):
        output_dir += '/'
    msg_to_save = {
        "batch_size": int(msg['batch_size']),
        "data_dir": msg['data_dir'][:-1] if msg['data_dir'].endswith('/') else msg['data_dir'],
        "epochs": int(msg['epoch']),
        "start_epoch": int(msg['start_epoch']),
        "fp16": 'fp16' in msg,
        "gpu_num": int(msg['gpu_num']),
        "model_size": msg['model_size'],
        "backbone_type": msg["backbone_type"] if "backbone_type" in msg else "origin",
        "output_dir": output_dir,
        "pretrained_weight_file": pretrained_file if use_pretrained_file else "no file selected",
        "save_each_epoch": 'save_each_epoch' in msg,
        "train_annotation_file": '%sannotations/%s' % (msg['data_dir'], msg['train_anno']),
        "train_dataset_path": '%s%s' % (msg['data_dir'], msg['train_dir']),
        "train_name": "output",
        "use_pretrained_weight": use_pretrained_file,
        "val_annotation_file": '%sannotations/%s' % (msg['data_dir'], msg['val_anno']),
        "val_dataset_path": '%s%s' % (msg['data_dir'], msg['val_dir']),
    }
    class_string = msg["classes"].replace("\r", "").replace("\t", "")
    save_string = ""
    for class_name in class_string.split("\n"):
        while class_name.endswith(" "):
            class_name = class_name[:-1]
        if len(class_name):
            save_string += f"{class_name}\n"

    if len(save_string) and os.path.isdir(msg["data_dir"]):
        save_string = save_string[:-1]
        open(os.path.join(msg["data_dir"], "classes.txt"), "w").write(save_string)
    try:
        gpu_choose = msg["gpu_choose"]
        gpu_choose.replace("；", ";").replace(" ", "").replace("\r", "")
        if len(gpu_choose):
            gpu_choose = [int(i)-1 for i in gpu_choose.split(';')]
            msg_to_save["gpu_choose"] = gpu_choose
    except:
        print("gpu编号选择填写有误！")
        pass
    # ret = ""
    # for name in msg_to_save:
    #     ret += "%s: %s<br />" % (name, msg_to_save[name])
    # return ret
    yaml.dump(msg_to_save, open('%ssettings.yaml' % output_dir, 'w'), yaml.Dumper)
    return create_html.jump2Html('/settings_list', ['success'], 1)


@app.route('/save_hyp', methods=["GET", "POST"])
def save_hyp():
    msg = get_webform()
    int_param = ["eval_interval", "no_aug_epochs", "print_interval", "warmup_epochs"]
    double_param = ["basic_lr_per_img", "min_lr_ratio", "momentum", "warmup_lr", "weight_decay"]
    # [print(name, msg[name]) for name in msg]
    if os.path.isdir('./settings/%s' % msg["name"]):
        old_msg = yaml.load(open('./settings/%s/hyp.yaml' % msg["name"]), yaml.Loader)
        old_msg['ema'] = False
        for name in msg:
            if name in old_msg:
                old_msg[name] = int(msg[name]) \
                    if name in int_param \
                    else float(msg[name]) \
                    if name in double_param \
                    else msg[name] \
                    if not name == 'ema' \
                    else True
        yaml.dump(old_msg, open('./settings/%s/hyp.yaml' % msg["name"], 'w'), yaml.Dumper)
        return create_html.jump2Html('/settings_list', 'success', 1)
    else:
        return create_html.jump2Html('/settings_list', 'error', 1)


@app.route('/delete_setting', methods=["GET", "POST"])
def delete_setting():
    msg = get_web_get()
    # print(msg)
    assert 'name' in msg
    return create_html.confirmDeleteSettingHtml(msg['name'])


@app.route('/confirm_delete', methods=["GET", "POST"])
def confirm_delete():
    msg = get_webform()
    # print(msg)
    if 'name' in msg and 'inputname' in msg:
        if msg['name'] == msg['inputname']:
            import shutil
            shutil.rmtree('./settings/%s' % msg['name'])
            return create_html.jump2Html('/settings_list', 'success', 1)
    return create_html.jump2Html('/delete_setting?name=%s' % msg['name'], 'failed', 1)


#############################################################################
# 开始训练  停止训练
@app.route('/start_train', methods=["GET", "POST"])
def start_train():
    msg = get_web_get()
    print(msg)

    is_training = os.path.exists('./settings/%s/pid' % msg['name'])
    if not is_training:
        commands = []

        start_dir = os.path.abspath('./settings/%s' % msg['name'])
        python_file = os.path.abspath('./start_train.py')
        interpreter = open("./run/interpreter.txt").read().split('\n')[0]

        commands.append('cd %s' % start_dir)
        commands.append('%s %s' % (interpreter, python_file))

        if not SYSTEM == "Windows":
            this_screen = Screen(name=msg['name'])
            for command in commands:
                this_screen.command(command)
        else:
            bat_file = os.path.join(start_dir, "train.bat")
            bat_string = ""
            for command in commands:
                bat_string += "%s\n" % command
            open(bat_file, 'w').write(bat_string)
            os.popen("start %s" % bat_file)

        return create_html.jump2Html('/settings_list', '训练成功开始！', 1)
    else:
        return create_html.jump2Html('/settings_list', '已经在训练了！！！', 1)


@app.route('/stop_train', methods=["GET", "POST"])
def stop_train():
    msg = get_web_get()
    # print(msg)
    is_training = create_html.is_training(msg['name'])
    if is_training:

        if not SYSTEM == "Windows":
            this_screen = Screen(name=msg['name'], create=False)
            this_screen.stop()
            create_html.stop_all_pid(msg['name'])
            this_screen.release()
        else:
            create_html.stop_all_pid(msg['name'])

        return create_html.jump2Html('/settings_list', 'success', 1)
    else:
        return create_html.jump2Html('/settings_list', '本来就没在训练！！！', 1)


#############################################################################
# 训练数据显示
@app.route('/train_log', methods=["GET", "POST"])
def train_log():
    msg = get_web_get()
    is_training = create_html.is_training(msg['name'])
    log_name = './settings/%s/output/train_log.txt' % msg['name']
    if os.path.exists(log_name):
        if "lines" in msg:
            lines = int(msg['lines'])
            all_lines = open(log_name, encoding="utf8").readlines()[-lines:]
            send_string = ""
            for line in all_lines:
                send_string += line  # .replace('\n', '<br \\>')

            return send_string + ("" if is_training else "\n[Not Training Now!]")
        else:
            params = {
                "SETTING_NAME": msg['name'],
                "LINE_LENGTH": msg["line_length"] if "line_length" in msg else 30
            }
            return render_template("train_log.html", web_name=create_html.get_web_name(), **params)
    else:
        return create_html.jump2Html('/settings_list', 'log file not found！！！', 1)


@app.route('/train_details', methods=["GET", "POST"])
def train_details():
    msg = get_web_get()
    # print(msg)
    # is_training = create_html.is_training(msg['name'])
    if 'name' in msg:
        if os.path.isdir('./settings/%s/output/epochs' % msg['name']):
            if len(glob('./settings/%s/output/epochs/*' % msg['name'])):
                length = msg['show_length'] if 'show_length' in msg else 30
                hyp_file_name = './settings/%s/hyp.yaml' % msg['name']
                hypmsg = yaml.load(open(hyp_file_name), yaml.Loader)
                print_interval = hypmsg["print_interval"]

                data = {
                    'SHOW_DATA_LENGTH': str(length),
                    'PRINT_INTERVAL': str(print_interval),
                    'SETTING_NAME': msg['name']
                }

                return render_template('train_table.html', web_name=create_html.get_web_name(), **data)
        return create_html.jump2Html('/settings_list', '从未训练过，无详情', 1)
    return create_html.jump2Html('/settings_list', '不要乱改request，小老弟', 1)


#############################################################################
# 来自训练详情页面的 request 请求
@app.route('/train_all_msg', methods=["GET", "POST"])
def train_all_msg():
    msg = get_web_get()
    if 'name' in msg:
        return json.dumps(create_html.extract_train_msg(msg['name']))
    return create_html.jump2Html('/settings_list', 'Wrong Request!', 1)


@app.route('/train_latest_msg', methods=["GET", "POST"])
def train_latest_msg():
    msg = get_web_get()
    if 'name' in msg:
        return create_html.extract_latest_train_msg(msg['name'])
    return create_html.jump2Html('/settings_list', 'Wrong Request!', 1)


@app.route('/eval_msg', methods=["GET", "POST"])
def val_msg():
    msg = get_web_get()
    if 'name' in msg:
        return json.dumps(create_html.extract_eval_msg(msg['name']))
    return create_html.jump2Html('/settings_list', 'Wrong Request!', 1)


@app.route('/pth_msg', methods=["GET", "POST"])
def pth_msg():
    msg = get_web_get()
    if 'name' in msg:
        return create_html.get_all_pth_files_list(msg['name'])
    return create_html.jump2Html('/settings_list', 'Wrong Request!', 1)


@app.route('/delete_pth', methods=["GET", "POST"])
def delete_pth():
    msg = get_web_get()
    if 'name' in msg and 'from' in msg:
        pth_path = './settings/%s/output/%s' % (msg['from'], msg['name'])
        if os.path.isfile(pth_path):
            os.remove(pth_path)
            if os.path.isfile(pth_path):
                return '删除失败！请检查文件权限！'
            else:
                return '删除成功!'
        return '错误：文件%s不存在' % msg['name']
    return '请求错误！'


#############################################################################
# 所有配置列表
@app.route('/settings_list', methods=["GET", "POST"])
def settings_list():
    if not os.path.exists('./run/interpreter.txt'):
        import sys
        open('./run/interpreter.txt', 'w').write(sys.executable.replace("\\", "/"))
    now_interpreter = open('./run/interpreter.txt').read()
    data = get_web_get()
    if "table" in data:
        return create_html.getSettingsList()
    return render_template("settings_list.html", web_name=create_html.get_web_name(), now_interpreter=now_interpreter)


# 登录页面
@app.route('/login', methods=["GET", "POST"])
def login():
    return login_base()


# 注销
@app.route('/logout')
def logout():
    return logout_base()


# 网站主页
@app.route('/', methods=["GET", "POST"])
def main():
    return index()


if __name__ == '__main__':
    os.makedirs('settings', exist_ok=True)
    args = make_parser().parse_args()
    print(args)
    if args.log_off:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
