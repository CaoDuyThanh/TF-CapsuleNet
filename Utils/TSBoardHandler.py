import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from StringIO import StringIO

class TSBoardHandler():
    def __init__(self,
                 _save_path):
        self.save_path = _save_path
        _summary_folder_path = self.save_path + 'summary/'
        self.summary_folder = tf.summary.FileWriter(_summary_folder_path)

        self.all_summary    = []
        self.merged_summary = None

    def save_graph(self,
                   _graph):
        _graph_folder_path = self.save_path + 'graph/'
        self.graph_folder  = tf.summary.FileWriter(_graph_folder_path, _graph)

    def summary(self,
                _name_scope,
                _func,
                **_args):
        with tf.name_scope(_name_scope):
            _summary = _func(**_args)
            self.all_summary.append(_summary)
            self.merged_summary = tf.summary.merge(self.all_summary)

    def log_scalar(self,
                   _name_scope,
                   _name,
                   _value,
                   _step):
        with tf.name_scope(_name_scope):
            _summary_value = tf.Summary.Value(tag          = _name,
                                              simple_value = _value)
            _summary = tf.Summary(value = [_summary_value])
            self.summary_folder.add_summary(_summary, _step)
            self.summary_folder.flush()

    def log_images(self,
                   _name_scope,
                   _name,
                   _images,
                   _step):
        with tf.name_scope(_name_scope):
            _im_summaries = []
            for _idx, _img in enumerate(_images):
                _str = StringIO()
                plt.imsave(_str, _img, format = 'jpg')

                # Create an Image object
                _img_sum = tf.Summary.Image(encoded_image_string = _str.getvalue(),
                                            height               = _img.shape[0],
                                            width                = _img.shape[1])

                # Create a Summary value
                _im_summaries.append(tf.Summary.Value(tag   = '%s/%d' % (_name, _idx),
                                                      image = _img_sum))

            # Create and write Summary
            _summary = tf.Summary(value = _im_summaries)
            self.summary_folder.add_summary(_summary, _step)
            self.summary_folder.flush()

    def log_histogram(self,
                      _name_scope,
                      _name,
                      _values,
                      _step,
                      _bins = 1000):
        with tf.name_scope(_name_scope):
            # Convert to a numpy array
            _values = numpy.array(_values)

            # Create histogram using numpy
            _counts, _bin_edges = numpy.histogram(_values, bins = _bins)

            # Fill fields of histogram proto
            _hist = tf.HistogramProto()
            _hist.min = float(numpy.min(_values))
            _hist.max = float(numpy.max(_values))
            _hist.num = int(numpy.prod(_values.shape))
            _hist.sum = float(numpy.sum(_values))
            _hist.sum_squares = float(numpy.sum(_values ** 2))

            # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
            # Thus, we drop the start of the first bin
            _bin_edges = _bin_edges[1:]

            # Add bin edges and counts
            for _edge in _bin_edges:
                _hist.bucket_limit.append(_edge)
            for _c in _counts:
                _hist.bucket.append(_c)

            # Create and write Summary
            _summary_hist = tf.Summary.Value(tag   = _name,
                                             histo = _hist)
            _summary = tf.Summary(value = [_summary_hist])
            self.summary_folder.add_summary(_summary, _step)
            self.summary_folder.flush()
