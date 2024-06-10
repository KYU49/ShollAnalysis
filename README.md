# TL;DR
Sholl Analysis for 16 bit tiff image.  
The following English is the machine translation from Japanese.  
日本語の説明も英語の下にあるよ。  

# Overview
Sholl Analysis is a morphological analysis method for neurons proposed by Sholl in 1953 [1], which has been widely used in many papers to date. Specifically, it involves drawing concentric circles at regular intervals around the cell body and counting the points where the circles intersect with neurite as intersections. When the distance from the cell body is taken as the x-axis and the number of intersections is taken as the y-axis, the graph typically shows an upward trend as the neurite branch out and a downward trend as the neurite terminate.  
In the past, this analysis was performed manually, but in recent years, it is more commonly measured using programs such as ImageJ Fiji plugins. However, these programs require binarization of the acquired images, making the measurement challenging in cases where there are a significantly large number of neurite, or when there is variation in the brightness of the neurite, such as in organoids. To address this, we have developed a program that takes the position on the concentric circles as the x-axis and the brightness as the y-axis, performs numerical differentiation using the Savitzky–Golay filter, and counts the number of peaks to determine the number of neurite. Compared to algorithms that require binarization, our program can accurately count dim axons and overlapping axons.

# Requirements
Python
OpenCV
pip install opencv-python
Numpy
conda install numpy

# Input/Output
The input requires a 16-bit tiff image with the cell(s) preferably centered. Also, there should be only one central cell or cell group.
The output consists of a tsv file containing the number of neurite per radius and a png file marked with the counted positions as neurite. Unbranched neurite are marked in blue (primary), those branching once are marked in green (secondary), and those branching twice or more are marked in red (tertiary).

# Usage
Set the parameters according to the instructions in shollAnalysis.py and execute.

# Usage in Papers
When using this program in papers or other publications, please make sure to include this GitHub URL and the date of downloading the file in the references section (to ensure reproducibility even with different versions).

[1] Sholl, D. A. (1953). Dendritic organization in the neurons of the visual and motor cortices of the cat. Journal of anatomy, 87(4), 387.

# 概要
Sholl Analysisは1953年にShollによって提案された[1]神経細胞の形態分析手法で、今年まで多くの論文で使用されている。具体的には細胞体を中心として一定間隔で同心円状に線を引き、その線と神経突起が交わった点をintersectionとして数える方法である。細胞体からの距離を横軸、intersectionの数を縦軸に取ると、概ね突起が分岐すると右上がりのグラフになり、突起の終端を迎えると右下がりのグラフになる。  
古くは手作業で数えていたが、近年ではImageJ Fijiのプラグインなどのプログラムで測定される場合が多い。しかし、これらのプログラムでは取得した画像を2値化する必要があり、organoidやなどにおいて突起の数が著しく多い場合や、突起の明るさにばらつきがある場合などでは測定が難しかった。そこで、同心円上の位置を横軸、輝度を縦軸に取り、Savitzky–Golay filterによって数値微分し、ピークピッキングによって本数を数えるプログラムを考案した。従来の2値化を必要とするアルゴリズムに比べ、輝度の低い軸索や重なっている軸索も正確にカウントすることができる。  

# Requirements
* Python
* OpenCV
```pip install opencv-python```
* Numpy
```conda install numpy```

# 入力・出力
入力には16 bit tiff画像が必要で、できるだけ細胞が中心にあることが望ましい。また、中心となる細胞(群)は1つであること。  
出力は半径ごとの神経突起の本数のtsvファイルと、神経突起としてカウントされた位置がマークされたpngファイルになる。分岐していない神経突起は青(primary), 1度分岐すると緑(secondary), 2度以上分岐したものは赤(tertiary)でマークされる。  

# 使い方
shollAnalysis.py内の指示に従ってパラメータを設定して実行。

# 論文などで使用する場合について
必ずReferenceにgithubのURLとファイルをダウンロードした日付(version違いで再現性が得られないことを防ぐため)を記載してください。  

# Reference
1. Sholl, D. A. (1953). Dendritic organization in the neurons of the visual and motor cortices of the cat. J. Anat. 87, 387–406.
