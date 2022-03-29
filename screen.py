import os
# os.system('sudo apt install screen -y')


class Screen(object):

    __opened = False

    def __init__(self, name: str or None = None, create=True):
        self.__screen_name = name
        self.__opened = not create
        if isinstance(self.__screen_name, str) and create:
            self.open(self.__screen_name)

    def open(self, name: str):
        self.__screen_name = name
        for line in os.popen('screen -ls').readlines():
            if line.startswith('\t') or line.startswith(' '):
                line = line.replace(' ', '').replace('\t', '')
                if '.' in line:
                    line = line.split('.')
                    screen_pid = line[0]
                    this_name = line[1].split('(')[0]
                    if this_name == self.__screen_name:
                        abs_name = "%s.%s" % (screen_pid, this_name)
                        os.popen('screen -x -S %s -X stuff "^C\n"' % abs_name)
                        os.popen('screen -x -S %s -X quit' % abs_name)

        os.popen('screen -dmS %s' % self.__screen_name)
        self.__opened = True

    def stop(self):
        if isinstance(self.__screen_name, str):
            for i in range(3):
                os.popen('screen -x -S %s -X stuff "^C\n"')

    def release(self):
        if isinstance(self.__screen_name, str):
            for i in range(3):
                os.popen('screen -x -S %s -X stuff "^C\n"')
            os.popen('screen -S %s -X quit' % self.__screen_name)
            self.__opened = False

    def command(self, cmd: str):
        os.popen('screen -x -S %s -X stuff "%s\n"' % (self.__screen_name, cmd))

    def isOpened(self):
        return self.__opened


if __name__ == '__main__':

    a = Screen('test')
    a.release()
