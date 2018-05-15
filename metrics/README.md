# metrics
## Root Mean Squared Logarithmic Error
- <https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError>
- <https://gist.github.com/Tafkas/7642141>

```
RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }
```

Some libraries don't provided RMSLE. Insted of it, Train log1p(target-variable) with RMSE and expm1(predicted).
