import pickle, io
from PIL import Image
import tensorflow as tf

def _bytes_feature(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _int64_feature(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def main():
    data = pickle.load(open('bscan_amd_imgs.p', 'rb'))
    writer = tf.io.TFRecordWriter('dataset.tfrecord')
    for fname, rec in data.items():
        img = Image.fromarray(rec['original_image'])
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        label_str = rec['label']
        transcript_bytes = label_str.encode()
        feat = {
            'image_raw': _bytes_feature(buf.getvalue()),
            'label': _int64_feature(1 if label_str=='A' else 0),
            'transcript': _bytes_feature(transcript_bytes),
        }
        ex = tf.train.Example(features=tf.train.Features(feature=feat))
        writer.write(ex.SerializeToString())
    writer.close()
    print("Wrote dataset.tfrecord")

if __name__ == '__main__':
    main()
