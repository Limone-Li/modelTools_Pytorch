import os
import sys
import time
import math


class ProgressBar(object):
    '''This is visual tools of training progress.

    Attributes:
        print(current, total, msg): msg is the message of your training progress.

    Examples:
        >>> a = ProgressBar()
        >>> a.start()
        >>> for i in range(11):
        >>>     time.sleep(1)
        >>>     a.print(i, 10, 'Progress', i / 10)
        >>> Out:
        >>> 100%[================>] Step:1s2ms | Total:11s36ms | hello
    '''
    __slots__ = ('begin_time', 'last_time')

    def __init__(self):
        super().__init__()
        self.begin_time = None
        self.last_time = self.begin_time

    def start(self):
        self.begin_time = time.time()
        self.last_time = self.begin_time

    def print(self, current, total, msg):
        if self.begin_time is None:
            raise Exception('Please use ProgressBar.start before print()')

        len_bar = 17

        begin_time = self.begin_time
        last_time = self.last_time
        end_time = time.time()

        total_str = self._format_time(end_time - begin_time)
        step_str = self._format_time(end_time - last_time)

        per = math.ceil((current / total) * len_bar)
        res = '{:3d}%'.format(int(current / total * 100)) + '[' + (
            '=' * max(per - 1, 0)) + '>' + '*' * (len_bar - per) + ']'

        res += ' Step:' + step_str + ' | Total:' + total_str
        res += ' | ' + msg

        self.last_time = end_time

        if current == total:
            print(res)
        else:
            print(res, end='\r')

    def flush(self):
        print()

    def _format_time(self, n):
        n = float(n)
        label = ['d', 'h', 'm', 's', 'ms']
        t = [0, 24, 60, 60, 1000]
        k = [0] * len(label)
        n = int(n * 1000)

        for i in range(len(label) - 1, 0, -1):
            k[i] = n % t[i]
            n = n // t[i]

        id = 3
        res = ''
        for i in range(len(label)):
            if k[i] > 0:
                res += str(k[i]) + label[i]
                id -= 1
            if id == 0:
                break
        return res
