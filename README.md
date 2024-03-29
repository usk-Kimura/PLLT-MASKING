# PLT-Masking
## 概要
PLT Score Maskingは、ドメイン適応のための新しいマスク戦略を提供します。この手法は、META-TARTANのコードベースを踏襲しつつ、本研究特有のスクリプトによる拡張を行っています。


[日本語論文](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=233524&item_no=1&page_id=13&block_id=8)
[日本語発表資料](https://github.com/usk-Kimura/PLT-MASKING/files/14751126/IFAT154.pdf)



### 特徴
**ドメイン適応**: 異なるドメイン間でのモデルの適用性を高めるための新しい手法。

**高い汎用性**: META-TARTANのフレームワークを基にしつつ、さらに柔軟なカスタマイズが可能。

**使いやすさ**: 既存のMETA-TARTANスクリプトから簡単に移行できるよう設計されています。

## はじめに

このプロジェクトは、META-TARTANのコードをベースとしています。META-TARTANのスクリプトを本プロジェクトのスクリプトに置き換えることで、独自のドメイン適応手法を実装しています。

### インストール
PLT-Maskingを使用するには、以下の手順に従ってください。

このリポジトリをクローンする:

'''
git clone https://github.com/yourusername/PLT-Score-Masking.git
'''

### 使用方法
PLT Score Maskingの基本的な使用方法は以下の通りです。META-TARTANのスクリプトを、本プロジェクト特有のスクリプトに置き換えて実行ファイルをrun_mlm_auxiliary_PLT-Masking.py に変更してください．

## 貢献
このプロジェクトへの貢献を歓迎します。イシューを開いたり、プルリクエストを送ったりする前に、まずはCONTRIBUTING.mdをご覧ください。

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。
