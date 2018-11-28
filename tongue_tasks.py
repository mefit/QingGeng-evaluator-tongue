from celery import Celery
import tempfile
import requests
import tongue_analysis
from PIL import Image
import json

app = Celery(__name__)
app.conf.update(broker_url = 'redis://localhost/1')

@app.task
def evaluate(user_id, id, url):
    try:
        with tempfile.NamedTemporaryFile() as f:
            with requests.get(url, stream = True) as r:
                for chunk in r.iter_content(chunk_size = 128):
                    f.write(chunk)

            with tongue_analysis.setup('./tongue.pb') as sess:
                result = tongue_analysis.analyze(f.name, sess = sess)

        with tempfile.NamedTemporaryFile() as f_m, tempfile.NamedTemporaryFile() as f_t:
            content = {'舌质':result['shezhi'], '舌苔':result['shetai']}

            Image.fromarray(result['mosaic_img']).save(f_m.name, 'jpeg')
            Image.fromarray(result['tongue_img']).save(f_t.name, 'jpeg')
            files = [
                ('files[]', ('mosaic', open(f_m.name, 'rb'), 'image/jpeg')),
                ('files[]', ('tongue', open(f_t.name, 'rb'), 'image/jpeg'))]

            requests.put(
                'https://qinggeng.app.aidistan.site/api/users/%s/evaluations/%s' % (user_id, id),
                headers = {'Authorization':'JCMXSorcbAAs8iVobTFGBxhX'}, files = files,
                data = {'completed':True, 'kind':'JSON', 'content':json.dumps(content)})
    except:
        content = {'status':'failed'}
        requests.put(
            'https://qinggeng.app.aidistan.site/api/users/%s/evaluations/%s' % (user_id, id),
            headers = {'Authorization':'JCMXSorcbAAs8iVobTFGBxhX'},
            data = {'completed':True, 'kind':'JSON', 'content':json.dumps(content)})
