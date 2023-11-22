#!/bin/bash

if [ -f "app.pid" ]
then
    pid=$(cat app.pid)
    kill $pid
    rm app.pid
    echo "服务已经停止，进程号为：$pid"
else
    echo "服务未启动"
fi
