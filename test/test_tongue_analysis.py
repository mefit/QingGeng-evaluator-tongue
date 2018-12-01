import sys
import pathlib
sys.path.append(str(pathlib.Path.cwd()))

import pytest
from tongue_analysis import setup as setup_ # to avoid name conflicts
from tongue_analysis import analyze


@pytest.fixture
def sess():
    sess = setup_('./tongue.pb')
    yield sess
    sess.close()


def test_session_loading(benchmark):
    def run():
        sess = setup_('./tongue.pb')
        sess.close()
    benchmark(run)

def test_analyze_image_0(sess, benchmark):
    def run():
        analyze('test_0.jpg', sess = sess)
    benchmark(run)

def test_analyze_image_1(sess, benchmark):
    def run():
        analyze('test_1.jpg', sess = sess)
    benchmark(run)

#
# To save the result returned by `analyze`:
#
#   result = analyze(tongue_image, sess = sess)
#   # dict() {'mosaic_img':打码图片, 'tongue_img':舌头图片, 'shezhi':舌质结果, 'shetai':舌苔结果}
#
#   Image.fromarray(result['mosaic_img']).save('mosaic.jpg', 'jpeg')
#