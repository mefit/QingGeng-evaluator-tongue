from celery import Celery
import tempfile
import urllib.request
import tongue_analysis

app = Celery(__name__)
app.conf.update(broker_url = 'redis://localhost/1')

@app.task
def evaluate(id, user_id, url):
    with tempfile.NamedTemporaryFile() as tf:
        with urllib.request.urlopen(url) as f:
            tf.write(f.read())

        with tongue_analysis.setup('./tongue.pb') as sess:
            result = tongue_analysis.analyze(tf.name, sess = sess)

        print(result)
        # JCMXSorcbAAs8iVobTFGBxhX
