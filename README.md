# modelTools_Pytorch
Some tools of Pytorch model

## Example:

Auto train a pytorch model.

<<<<<<< HEAD

=======
    model = models.VGG(make_layers(cfgs['A']), 10)
    model_name = 'vgg'
    file_path = '../tmp/vgg.pth'
    mt = ModelTool(model, model_name, file_path)
    mt.auto_train(train_loader, test_loader, epoch_max=20, save_epoch=5)
>>>>>>> 8b09babbc7ce4529ae6936416c3e61dcae9b2726

## SP. Command-line

__Rebuild sphinx doc.__

    cd docs
    sphinx-apidoc -o source '../tools' -f
    make clean
    make html

__Test Code.__

    python -m unittest discover -s './test' -p '*_test.py'

