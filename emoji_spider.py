import requests
import re


def get_page_url(keyword, page_index, images_per_page):
    pn = (page_index - 1) * images_per_page
    # 百度图片搜索url特征分析
    page_url = "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=" + keyword + "&pn=" + str(
        pn) + "&ct=&ic=0&lm=-1&width=0&height=0"
    return page_url


def get_images_url(page_url):
    html = requests.get(page_url).text
    # 正则表达式获取图片url
    images_url = re.findall('"objURL":"(.*?)",', html, re.S)
    return images_url


if __name__ == '__main__':
    keyword = '仓鼠表情'
    # 可根据搜索页数下载
    page_start = 1
    page_end = 15
    images_per_page = 20
    images_url = []
    # 存入图片列表
    for page_index in range(page_start, page_end + 1):
        page_url = get_page_url(keyword, page_index, images_per_page)
        images_url += get_images_url(page_url)[0:images_per_page]
    cnt = 0
    print('Downloading...')
    # 图片下载和存储
    for each in images_url:
        try:
            image = requests.get(each, timeout=10)
        except OSError:
            continue
        string = './Spider/' + keyword + '.' + str(cnt + 1) + '.jpg'
        fp = open(string, 'wb')
        fp.write(image.content)
        fp.close()
        cnt += 1
        # 综合考虑每个种类下载两百张
        # 后期图片要经过处理剔除gif和损坏的图片
        if cnt >= 200:
            break
