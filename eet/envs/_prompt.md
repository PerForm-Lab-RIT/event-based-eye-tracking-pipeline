how do I design a dataset generator (in @base.py  ) that generate stream of event data similar to how @replay.py does it? there are several things to note:
- The datagen has its own assemble batch and assemble sequence
- The datagen should be reading, loading and processing data into some buffer in parallel with sampling so the user can sample from it
- The datagen can support epoch (meaning: stop iteration)
- play nicely with the stream api in @streams.py 
- in this dataset, online is default, meaning that we always prioritize closest loaded data. Because often time, the loaded data is sequential
