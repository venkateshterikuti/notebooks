import os, urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
os.makedirs('data', exist_ok=True)
path = 'data/tinyshakespeare.txt'
if not os.path.exists(path):
    print('Downloading TinyShakespeare...')
    urllib.request.urlretrieve(url, path)
    print('Saved to', path)
else:
    print('Already exists:', path)
