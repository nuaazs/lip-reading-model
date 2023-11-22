import requests

url = 'http://127.0.0.1:9785/lip-reading'
data = {
    'raw_video_path': '/home/zhaosheng/Documents/lip-reading-model/test/input_videos/gfb.mp4',
    'start': '0.38',
    'end': '1.7',
    'sid': 'xyk'
}

response = requests.post(url, data=data)
print(response.text)
