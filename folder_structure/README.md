# folder structure
いろいろあるので、とりあえず使ってみて考える

## cookiecutter for kaggle
<https://github.com/uberwach/cookiecutter-kaggle>

- 参考元はデータサイエンス用の cookiecutter project <https://github.com/drivendata/cookiecutter-data-science>
- 以下、フォルダ構成
    - ref: codeblock with details/summary
    - [Qiitaのdetails,summary要素とMarkdownの相性調査](https://qiita.com/khsk/items/a002630c034c98edc9d5)

<details><summary>フォルダ構成</summary><div>

```
├── LICENSE
├── Makefile           <- Makefile with commands that perform parts of the processing pipeline
├── README.md          <- The top-level README
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── Dockerfile         <- Dockerfile, alternative approach to manage environment
│                         more interesting if using non-Unix
├── submissions        <- Directory to keep submissions
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions for submissions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
</div></details>

## machine learning competition with cookiecutter
<https://github.com/ksomemo/ml-competition-template-titanic>

### reference
<https://github.com/upura/ml-competition-template-titanic>

- [【Kaggleのフォルダ構成や管理方法】タイタニック用のGitHubリポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
- これの参考元は,　<https://github.com/flowlight0/talkingdata-adtracking-fraud-detection>

## discussion
<https://www.kaggle.com/general/4815>

## for team
- [Kaggleのための小規模なMLプロジェクトで頑張った話: MLプロジェクト設計編](https://qiita.com/mocobt/items/c16fa9f662257556425a)
- repository: <https://github.com/JapaneseTraditionals/kaggle_Malware>

<details><summary>フォルダ構成</summary><div>

```
.
├── config              # 設定ファイルのjsonを置くdirectory(後述)
├── data                # あまり見ないデータ一覧．基本的に共有はしない
│   ├── external        # Kaggleで急に上がってくるsubmission以外のcsvを置くDirectory
│   ├── features        # 特徴量を保存するdirectory
│   ├── model           # modelを保存するdirectory
│   ├── oof             # 検証用データに対するラベルと予測結果が記録されたcsvを保存するdirectory
│   ├── submit          # Submission用csvを保存するdirectory
│   ├── subset          # trainを分割した結果を保存しておくdirectory
│   └── validations     # trainの各データをどのfoldに入れるのかを定義するファイル(拡張子任意)を保存しておくdirectory
├── features            # 特徴量を生成するコンペ特有なコード(.py)を置くdirectory
├── importance          # 特徴量のimportanceをcsvとして保存しておくdirectory
├── input               # Kaggleから落としたcsv全般を保存しておくdirectory
├── log                 # log全般
│   ├── main            # コード全体の進捗を表すlogを保存しておくdirectory
│   └── train           # Classifierの学習進捗を表すlogを保存しておくdirectory
├── notebook            # notebook(.ipynb)全般 
│   ├── deprecated      # 使う価値のないipynbを追いやるdirectory
│   └── eda             # EDAを行ったipynbを置くdirectory
├── src                 # コード全般(.py)を置くdirectory
│   └── classifier      # LightGBMやCatBoost，NNなどのコード全般を置くdirectory 
└── tmp                 # どうしても一時的になにかを置きたいときに使うdirectory(gitで共有はしない)
```
</div></details>

## kaggle 以外でのデータサイエンスでの構造
なぜそのフォルダ構成にするか？についての言及やリンクが多くて参考になる

- [データサイエンスプロジェクトのディレクトリ構成どうするか問題](https://takuti.me/note/data-science-project-structure/)
- [データ分析をするときのフォルダ構成をどうするのか問題について](https://www.st-hakky-blog.com/entry/2017/03/24/140738)
