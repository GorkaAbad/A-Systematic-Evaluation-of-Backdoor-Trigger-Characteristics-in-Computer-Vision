unzip tiny-imagenet-200.zip
rm -r ./tiny-imagenet-200/test
python3 val_format.py
find . -name "*.txt" -delete
cp -r tiny-imagenet-200 tiny-224
python3 resize.py