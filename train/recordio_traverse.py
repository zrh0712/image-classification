import mxnet as mx
import sys

data_iter = mx.io.ImageRecordIter(
  path_imgrec=sys.argv[1], # The target record file.
  preprocess_threads=32,
  data_shape=(3, -1, -1), # Output data shape; 227x227 region will be cropped from the original image.
  batch_size=128, # Number of items per batch.
  resize=256 # Resize the shorter edge to 256 before cropping.
  # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
  )
# You can now use the data_iter to access batches of images.

i=0
while(True):
    i+=1
    try:
        batch = data_iter.next() # first batch.
        images = batch.data[0]
        print '{}th batch: {}'.format(i,images.shape)
    except StopIteration:
        print 'traverse success'
        break
