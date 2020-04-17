**Dependancy**

This implementation should be run with Python 3.x and Pytorch 0.4.0.

**Train**
```
1. python train.py
2. python train.py --finetune
```

**CONFIGURATION**
```
--testset   A OR B OR C
--sample_shape_number 3 #how many shapes do you want to sample parts from
```

Only need to change configuration before run demo.

**Demo**
```
python correspondence.py
python mergeTest.py
```

For example, results will be saved in ./result/A/
