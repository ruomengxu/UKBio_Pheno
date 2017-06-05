## How to run

### Data Cleaning

- Create jar

```
sbt package
```

- Upload data folder and jar to the server
- Run jar on the server (under /data/username/)

```
spark-submit xxxxx.jar
```

### Feature construction & Modeling & Visualization

- Upload scripts folder to the sever (/data/username)
- Run code

```
python main.py
```

