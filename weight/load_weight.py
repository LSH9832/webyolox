import requests
import threading

lock = threading.Lock()
now_size = 0

def Handler(start, end, url, filename, file_size):
    global now_size
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:68.0) Gecko/20100101 Firefox/68.0",
        'Range': 'bytes=%d-%d' % (start, end)
    }
    with requests.get(url, headers=headers, stream=True) as r:
        with open(filename, "r+b") as fp:

            fp.seek(start)
            var = fp.tell()
            for c in r.iter_content(chunk_size=1024):
                if c:
                    fp.write(c)
                    lock.acquire()
                    now_size += 1024
                    print("\r下载进度: %.2f %%" % (100*now_size/file_size), end="")
                    lock.release()


def download(url, file_name, num_thread=10):
    global now_size
    try:
        r = requests.head(url)
        file_size = int(r.headers['content-length'])
    except:
        print("检查URL，或不支持对线程下载")
        return 0
    if file_size < 1024:
        return -1
    fp = open(file_name, "wb")
    fp.truncate(file_size)
    fp.close()
    part = file_size // num_thread

    for i in range(num_thread):

        start = part * i
        if i == num_thread - 1:
            end = file_size
        else:
            end = start + part

        t = threading.Thread(target=Handler, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name, "file_size": file_size})
        t.setDaemon(True)
        t.start()

    # 等待所有线程下载完成
    main_thread = threading.current_thread()

    for t in threading.enumerate():
        if t is main_thread:
            continue
        t.join()
    now_size = 0
    return 1

# def download_single():



if __name__ == '__main__':
    pths = {
        "yolox_nano.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/nano_backbone.pth",
        "yolox_tiny.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/tiny_backbone.pth",
        "yolox_s.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/s_backbone.pth",
        "yolox_m.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/m_backbone.pth",
        "yolox_l.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/l_backbone.pth",
        "yolox_x.pth": "https://github.com/LSH9832/webyolox/releases/download/0.2.0/x_backbone.pth",
    }

    for pth in pths:
        print("开始下载 %s" % pth)
        flag = download(
            url=pths[pth],
            file_name=pth
        )
        if flag == 1:
            print("\n%s 下载完成\n" % pth)
        elif flag == -1:
            print("%s下载失败！请检查网络连接\n" % pth)
