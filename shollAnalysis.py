import cv2
import numpy as np
import math
from dataclasses import dataclass

# Requirements
# pip install opencv-python
# pip install numpy

############################
#
# Set the below paramerters
# この下のパラメータをいじること
#
############################
# Ignore a certain distance from the center. It doesn't have to be 0 since it can be removed later. Unit: μm; 
# 中心から一定距離を無視する。どうせ後で消せば良いから、0でなければ良い。単位はμm
start = 210
# How many μm to divide by? If counting branch order, it should be relatively small; 
# 何μmずつに区切るか。Branch orderを数えるなら、ある程度小さく。
gap = 2
# How many pixels to reference using the SG method? Default: 4; 
# SG法で前後何pixelを参照するか？Default: 4
N_SG = 4
# What degree of equation to approximate with the SG method? It must be less than N_SG * 2 + 1 and greater than or equal to the order of differentiation. Default: 4; 
# SG法で何次の方程式で近似するか。N_SG * 2 + 1未満であることが必要で、微分階数以上。Default: 4 
DIM = 4 
# Threshold. Exclude extremely dark pixel (< THRESHOLD).
# 輝度の閾値。極端に暗い点(< THRESHOLD)は除く
THRESHOLD = 1000 

# The file name of the .tif file
# .tifファイルのファイル名
tif = "./ShollAnalysis/projection.tif"
# The location to save the results (in this case, save in the same directory)
# 結果を保存する場所(この場合は同じディレクトリに保存)
outputDir = "./ShollAnalysis"   

# How many μm is 1 px? If a 1000 px x 1600 px image corresponds to 325 μm x 520 μm, it is 0.325
# 1 pxが何μmか？1000 px x 1600 pxの画像が325 μm x 520 μmなら0.325
pixelSize = 0.325

############################
#
# End of the paramertes
# いじる必要があるのはここまで
#
############################

@dataclass  # dataclassはmutable
class Pixel:
    value: int              # その座標の輝度
    x: int                  # x座標
    y: int                  # y座標
    isAxis: bool = False    # 軸索か否か。
    parent = None           # この軸索に繋がるより内側の円のPixel
    childCount: int = 0     # この軸索に繋がるより外側の円のPixelの数
    branchOrder: int = 1    # この軸索のBranch order。細胞体から直接生えているのは1、分岐するたびに1増える

@dataclass
class Intersection:
    primary: int = 0
    secondary: int = 0
    tertiary: int = 0
    def sum(self):
        return self.primary + self.secondary + self.tertiary


def main():
    # ファイルから画像読み込み
    countIntersection: list[Intersection]
    radi: list[int]
    radi, countIntersection, image_rgb = shollAnalysis(tifFileName = tif, pixelSize = pixelSize)
    
    # 各円周上の個数を書き出し
    with open(outputDir + "/shollAnalysis.tsv", mode="a", encoding="utf-8") as f:
        print("Distance from Sphere Center /μm" + "\t" + "\t".join(map(str, radi)), file=f)
        print("# " + tif, file=f)
        print("primary" + "\t" + "\t".join(map(str, [intersection.primary for intersection in countIntersection])), file=f)
        print("secondary" + "\t" + "\t".join(map(str, [intersection.secondary for intersection in countIntersection])), file=f)
        print("tertiary" + "\t" + "\t".join(map(str, [intersection.tertiary for intersection in countIntersection])), file=f)
        print("total" + "\t" + "\t".join(map(str, [intersection.sum() for intersection in countIntersection])), file=f)
    cv2.imwrite(outputDir + "/sholl.png", image_rgb)

def shollAnalysis(tifFileName, pixelSize, centerX = -1, centerY = -1):
    # ファイルから画像読み込み
    img = cv2.imread(tifFileName, cv2.IMREAD_UNCHANGED)
    # 背景などの前処理
    preTreated = preTreat(img)

    # 画像サイズの取得
    pixelHeight, pixelWidth = preTreated.shape[:2]

    # px単位からum単位に変換
    umWidth = int(pixelWidth * pixelSize)
    umHeight = int(pixelHeight * pixelSize)

    # 中心座標を指定。-1が指定されていた場合は画像中心
    if(centerX == -1):
        centerW = pixelWidth / 2
    else:
        centerW = centerX
    if(centerY == -1):
        centerH = pixelHeight / 2
    else:
        centerH = centerY

    shortestRadi = int(
        min(
            umWidth - centerW * pixelSize,  # 設定した中心から右端までの長さ
            centerW * pixelSize,            # 左端から設定した中心までの長さ
            umHeight - centerH * pixelSize, # 設定した中心から下端までの長さ
            centerH * pixelSize             # 上端から設定した中心までの長さ
        ) / 2)
    # 同心円の半径のリストを作成。pxではなく、um単位; 中心座標から画像端までの長さを確認し、画面の端すれすれまで円を作成。
    radi = [i for i in range(start, shortestRadi, gap)]
    
    circle: list[list[Pixel]] = [[] for _ in range(len(radi))]  # 各円に所属するピクセルの情報をPixel data classで保存。
    countIntersection: list[Intersection] = []       # 各半径ごとに軸索と認識された数を保存する配列

    # 同心円の半径リストから半径の値を取得し、各半径に属するpixelの座標を重複なく取得
    for j, r in enumerate(radi):
        rPx = int(r / pixelSize)
        lastX = -1
        lastValue = -1
        firstX = -1
        firstY = -1
        for theta in range(int(2 * math.pi * rPx)): # 円周の長さは2πr[px]
            t = float(theta) / rPx   # 1週で2πになるようにrPxで割る
            x = round(centerW + rPx * math.cos(t))
            value = round(centerH + rPx * math.sin(t))
            if (lastX != x and lastValue != value):
                # 0, 90, 180, 270°付近で飛び飛びになるため、補間する
                if lastX != -1 and lastValue != -1: # 最初の1 px以外は前のpixelの座標を使って間の座標も登録
                    for xy in lineInterpolation(lastX, lastValue, x, value):
                        if xy[1] != lastValue or xy[0] != lastX:
                            circle[j].append(
                                Pixel(
                                    value = preTreated[xy[1], xy[0]],
                                    x = xy[0],
                                    y = xy[1]
                                )
                            )
                        lastX = xy[0]
                        lastValue = xy[1]
                else:   # 最初の1マス
                    circle[j].append(
                        Pixel(
                            value = preTreated[value, x],
                            x = x, 
                            y = value,
                        )
                    )
                    firstX = x
                    firstY = value
                lastX = x
                lastValue = value
        if lastX != firstX or lastValue != firstY:  # 最後と最初を繋ぐ
            for xy in lineInterpolation(lastX, lastValue, firstX, firstY):
                if (xy[1] != lastValue or xy[0] != lastX) and (xy[1] != firstY or xy[0] != firstX): # 重ならないように調整
                    circle[j].append(
                        Pixel(
                            value = preTreated[xy[1], xy[0]],
                            x = xy[0],
                            y = xy[1]
                        )
                    )
                lastX = xy[0]
                lastValue = xy[1]

    # SG法で数値微分(https://qiita.com/Cartelet/items/2c6001f6cda163ee61c4), 3階微分が負から正になったところをピークとしてピッキング(https://www.toyo.co.jp/mecha/faq/detail/id=2749)
    image_8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    for c in circle:
        loopedX = [i for i in range(len(c) + 2 * N_SG)]   # SG法で数値微分する際のx値。等間隔であれば値は問わないため、単に0から順に並べるだけ。
        originalValues = [pixel.value for pixel in c]
        loopedValues = originalValues[-N_SG:] + originalValues + originalValues[:N_SG]  # SG法で削られる分を補完(円状に座標を取っているため、ループさせた値を取得すれば問題ない)
        _, dif3values = SG(loopedX, loopedValues, N_SG, DIM, 3)  # 3階微分の値

        lastValue = loopedValues[-1]   # 3階微分が負から正に変わるところを取りたいため、1つ前の値を保存する用。円状なので、最初の値と比較するために最後の値を入れておく。

        for j, value in enumerate(dif3values):
            if lastValue < 0 and 0 < value and originalValues[j] > THRESHOLD: # 三階微分が負から正に変わり、極端には暗くないところ
                # 1つ前の座標と、今回の座標のどちらがよりピークトップに近いか判定
                if abs(lastValue) < abs(value):
                    c[j - 1].isAxis = True                          # 軸索判定されたことを記録(何度分岐しているか判定するためのグラフ計算に使用)
                else:
                    c[j].isAxis = True                              # 軸索判定されたことを記録(何度分岐しているか判定するためのグラフ計算に使用)
                # countIntersection[-1] += 1  # 軸索の本数を保存(軸索のbranch orderを考えない場合はここで追加していたが、コメントアウト)
            lastValue = value   # 1つ前の値との比較をするため、今回の阿智を保存しておき、次の値と比較

    # ここからグラフ理論で分岐情報を計算
    axisListEachCircle: list[list[Pixel]] = [
        [px for px in c if px.isAxis] for c in circle
    ]   # 軸索判定されているPixelのみを取得
    
    # まず、外側の円から順に軸索として認識された点を1つずつ選び、
    # 1つ内側の円の中で一番近い軸索として認識された点を取得。
    # また、自分にアサインされている外側の円の点の数を保存。
    lastC = None    # 外側の円を保存しておく
    for c in reversed(axisListEachCircle):
        if lastC is None:
            lastC = c
            continue
        for outerPx in lastC:
            distance = 1000000000   # 適当に大きな数。現在評価した中で一番近いPixelとの距離の2乗。
            for innerPx in c:
                newDistance = (innerPx.x - outerPx.x) ** 2 + (innerPx.y - outerPx.y) ** 2
                if newDistance < distance:
                    outerPx.parent = innerPx
                    distance = newDistance
            outerPx.parent.childCount += 1
        lastC = c
    # 内側の円から順に、parentにアサインされている軸索の数を数える。
    # parentにアサインされている軸索が2以上なら、分岐したとみなして、自分のbranch orderをparentより1大きくする。
    flag = False    # 1つ目をスキップするためだけ
    for c in axisListEachCircle:
        if flag:
            currentIntersection = Intersection()
            countIntersection.append(currentIntersection)
            for px in c:
                parent: Pixel = px.parent
                if parent.childCount > 1:
                    px.branchOrder = parent.branchOrder + 1
                else:
                    px.branchOrder = parent.branchOrder
                
                # 色はO’Neill, K. M. et al. Front. Cell. Neurosci. 2015, 9, 1.に合わせてみた。
                if px.branchOrder == 1:     # primaryならblue(cv2はrgbじゃなくてbgr)
                    image_rgb[px.y, px.x] = (255, 0, 0)         # ピークの座標に点をつける(cv2の画像の座標は(縦, 横), つまり(y, x)で指定)
                    currentIntersection.primary += 1
                elif px.branchOrder == 2:   # secondaryならgreen
                    image_rgb[px.y, px.x] = (0, 255, 0)         # ピークの座標に点をつける(cv2の画像の座標は(縦, 横), つまり(y, x)で指定)
                    currentIntersection.secondary += 1
                else:                       # tertiary以降ならred
                    image_rgb[px.y, px.x] = (0, 0, 255)         # ピークの座標に点をつける(cv2の画像の座標は(縦, 横), つまり(y, x)で指定)
                    currentIntersection.tertiary += 1
        else:
            for px in c:
                image_rgb[px.y, px.x] = (255, 0, 0)         # ピークの座標に点をつける(cv2の画像の座標は(縦, 横), つまり(y, x)で指定)
            countIntersection.append(
                Intersection(primary = len(c))
            )
            flag = True

    return radi, countIntersection, image_rgb


def preTreat(origin):
    img = origin.copy()

    # バックグランド除去
    img = subtract_background(img, radius=30, light_bg=False)
    return img

def subtract_background(image, radius=50, light_bg=False):
        if light_bg:
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, circle_kernel(radius))
        else:
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, circle_kernel(radius))

def circle_kernel(radius):
    c = math.ceil(radius - 1)
    s = c * 2 + 1
    return np.clip(radius - np.sqrt(np.sum((np.stack((
        np.tile(np.arange(s), (s, 1)),
        np.repeat(np.arange(s), s).reshape((-1, s))
    )) - c) ** 2, axis=0)), 0, 1).astype("uint8")

# (x1, y1)と(x2, y2)を結ぶ直線上にある点を全て整数値で取得
def lineInterpolation(x1, y1, x2, y2):
    coordinates = []
    # 差分を取得
    dx = x2 - x1
    dy = y2 - y1
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    # 大きい方の差分に合わせてループを回す
    max_diff = max(abs_dx, abs_dy)
    # ステップサイズを決定
    step_x = dx / max_diff
    step_y = dy / max_diff
    
    for i in range(max_diff + 1):
        # x座標を取得
        x = round(x1 + i * step_x)
        # y座標を取得
        y = round(y1 + i * step_y)
        coordinates.append((x, y))
    return coordinates

def SG(x, y, N, m, d=0):
    dx = x[1] - x[0]
    X = (np.c_[-N:N+1] * dx) ** np.r_[:m+1]
    C = np.linalg.pinv(X) # (X.T X)^-1 X.T
    x_ = x[N:-N]
    y_ = np.convolve(y[::-1], C[d], "valid")[::-1]
    return x_, y_

if __name__ == "__main__":
    main()