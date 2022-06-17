训练所需内存：6GB
训练时间：2h 50min
预测时间：15min

执行流程:
1、安装依赖环境：
pip install -r requirements.txt

2、将数据文件 comp_2022_all_rank_b_data.tsv 放于脚本同一目录下，执行模型训练脚本 train_main.sh，生成模型文件 model.txt 和 验证集mape文件 mape_val.csv
sh train_main.sh

3、执行模型预测脚本pred_main.sh，生成预测文件submit.tsv
sh pred_main.sh

