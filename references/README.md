# references

## kaggle全般
- <https://github.com/nejumi/kaggle_memo>
- <https://github.com/amaotone/kaggle-memo>
- <https://kaggler-ja-wiki.herokuapp.com/>

## missing value
- [データ分析プロセス Useful R 第3章　前処理・変換 3.2　欠損値への対応](http://www.kyoritsu-pub.co.jp/bookdetail/9784320123656)

## feature selection
- <http://scikit-learn.org/stable/modules/feature_selection.html>
  - low variance
    - include: `n_unique=0`
  - using SelectFromModel
    - Lasso: coef_
    - Tree: feature_importances_
      - value 0 columns: not used
- <https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Usage.ipynb>
  - missing value: threshold ?
  - multi collinearity: threshold ?

## feature engineering
preprocessing も混じっているので整理する

- <https://www.slideshare.net/HJvanVeen/feature-engineering-72376750>
- [自分のメモpandasで複雑な集約](https://ksomemo.github.io/contents/qiita/pandas%20window%20function%E9%96%A2%E9%80%A3%E3%81%A80.18.1%E3%81%AE%E6%A9%9F%E8%83%BD%E8%A9%A6%E3%81%97%E3%81%9F(+%E6%9D%A1%E4%BB%B6%E4%BB%98%E3%81%8Dgroupby%E9%9B%86%E8%A8%88.html)
 - SQLでやってしまってもいいと思う

### categorical feature
<http://contrib.scikit-learn.org/categorical-encoding/index.html>

- 機械学習するdatasetへの集約時
  - `df.groupby("id")["cat_col"].agg(["size", "count", "nunique"]).add_prefix("cat_col_")`
  - sizeは割合作成時に利用するかもしれないので集計しておく
  - category columnsの組合せで集約する
- Label Encoding
  - `from sklearn.preprocessing import LabelEncoder`
- One Hot Encoding
  - `pd.get_dummies`
  - TODO: sholud use `drop_first=True` for tree ?
- Taget Encoding
  - `target 0/1 or num`
    - join or map, `df.groupby("cat_col")["target"].mean()`
    - concat, `df.groupby("cat_col")["target"].transform("mean")`
  - `for multi classes`: TODO
- Count Encoding
  - `df["cat_col"].value_counts(dropna=False)`
  - `dropna=False` しても null ではjoinできないので、fillna() しておくこと
- Count Rank Encoding
  - `df["cat_col"].value_counts().rank(ascending=False, method="xxx")`

### numerical feature
- 機械学習するdatasetへの集約時
  - `df.groupby("id")["num_col"].describe(percentiles=[]).add_prefix("num_col_")`
  - count, mean, std, min, 50%, max
  - 中央値限定にして(残ってしまって25,75%を含まないだけ)無駄な集計していないが、中央値も不要であれば別のやり方で処理速度も効率良くできる　
- binarize: 解釈しやすくする
  - `df["col"] >= threshold`
  - 複数カラムでのbinarize からcrossの組合せで one hot encoding
- 変換
  - 差分: `利益 = 売上 - コスト`
  - 平均: `単価 = 購入金額 / 購入回数`
  - 割合: `直帰率 = サービス利用日数 / サービス利用開始からの日数`
  - log: `x=0`のとき注意, 0考慮の `log(x+1)` などで対処
  - 差分の大きさ: abs, `pow -> sqrt`
  - 交互作用項 ?
    - この文脈でそう言ってよいのか不明
  - scale: standard, min-max
- [連続値データの離散化(R Advent Calendar 2013)](http://d.hatena.ne.jp/sfchaos/20131208/p1)
  - 若干違う
  - 等間隔(binning), 指定間隔
    - `pd.cut(x, bins=20)`
    - `pd.cut(x, bins=range(0, 20+1, 5))`
  - 分位数, quantile:
    - `pd.qcut(x, 4) == pd.qcut(x, [0, 0.25, 0.5, 0.75, 1])`
  - 離散化してからcategorical featureとしての変換

### 時系列を考慮
- 移動平均
- 直近の行動の集計: 1, 7, 30, 180, 365など期間指定
- 前半・中盤・後半など等間隔の期間分割
  - 期間中の集約値でもよいが、行動割合への正規化
- 基準日時と対象日時の日時差分による数値化
  - 登録からの初行動までの日数
  - 最終月末から直近行動までの日数(何日行動していない)

### image
- packages
  - opencv
  - skimage
  - face detection
    - dlib
    - face_recognition
- gray scale
- HSV, H:色彩は使いづらい
- resize: 処理時間短縮
- convolution: apply filter
- PCA ?
- text extracton ?

### movie
- packages, library
  - ffmpeg
    - cut検出
    - opencv等のpackageで利用している
  - opencv
  - skvideo

### natual language
- 形態素解析
- `n-gram`
- TF-IDF
  - Term Frequency
  - Inverse Document Frequency
- Bag-of-Words
  - One Hot Encoding
- embedding: 分散表現
  - word2vec

## ensemble
- <https://mlwave.com/kaggle-ensembling-guide/>
- 複数modelによる予測, どのmodelを重要とするかは重み付けを行う
  - tree系, NN系, etc. などのように異なるalgorithmを使っているといいらしい

### voting (class label)
- hard: 多数決
- soft: 各クラスに属する予測確率値のうち最大のクラス? TODO
  - [scikit-learnモデルのVotingとキャッシング](http://segafreder.hatenablog.com/entry/2016/08/14/004633)

### bagging
- `Bootstrap` + `AGGregatING`
- `bootstrap sampling` による非復元抽出によるsample作成

#### random forest
- random sampling columns
- estimator is a decision tree
- `Out-of-Bag`
  - [【機械学習】OOB (Out-Of-Bag) とその比率](https://qiita.com/kenmatsu4/items/1152d6e5634921d9246e)
- feature importances
  - [【python】ランダムフォレストのOOBエラーが役に立つか確認](https://hayataka2049.hatenablog.jp/entry/2018/07/30/020209)
  - [Random Forestで計算できる特徴量の重要度](http://alfredplpl.hatenablog.com/entry/2013/12/24/225420)
  - [](http://shindannin.hatenadiary.com/entry/2015/04/25/142452)
- decision tree
  - [決定木アルゴリズムを実装してみる](http://darden.hatenablog.com/entry/2016/12/15/222447)
  - [決定木 変数重要度 / Decision Tree Variable Importance](https://speakerdeck.com/sasakik/decision-tree-variable-importance?slide=17)
  

### boosting
TODO

### stacking
TODO

## data augumentation
### image
TODO

## imbalanced data
<http://imbalanced-learn.org/en/stable/>

### over sampling
実際には存在しないデータなので、説明できないと業務で使えないのでつらい

- kNN, SVM を利用した方法が存在する
- SMOTE: TODO

### under sampling
- Bagging + Under Sampling
  - Bootstrap Samplingして得たdatasetに対してUnder Sampling
  - その後Decision Treeで学習し、aggregating

## hyperparameter optimization
- Grid Search
- TODO

## cross validation
- numerical target: K-fold
- categorical target: Stratified K-Fold
- time series: `sklearn.model_selection.TimeSeriesSplit`
  - `max_train_size=30`: train size is approximate 30.
    - |train|test-|-----|-----|
    - |-----|train|test-|-----|
    - |-----|-----|train|test-|
    - |-t1--|-t2--|-t3--|-t4--|
  - `max_train_size=None`: train is everything before test.
