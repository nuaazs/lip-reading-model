#!/bin/bash

# 启动 gunicorn
gunicorn -w 4 -b 0.0.0.0:9785 app:app --daemon --pid app.pid

echo "服务已经启动，进程号为：$(cat app.pid)"
