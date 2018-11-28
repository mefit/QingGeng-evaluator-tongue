from celery import Celery
import tempfile
import requests
import tongue_analysis
import json

app = Celery(__name__)
app.conf.update(broker_url = 'redis://localhost/1')

@app.task
def evaluate(user_id, id, url):
    with tempfile.NamedTemporaryFile() as f:
        with requests.get(url, stream = True) as r:
            for chunk in r.iter_content(chunk_size = 128):
                f.write(chunk)

        with tongue_analysis.setup('./tongue.pb') as sess:
            result = tongue_analysis.analyze(f.name, sess = sess)

        content = {'舌质':result['shezhi'], '舌苔':result['shetai']}
        requests.put(
            'https://qinggeng.app.aidistan.site/api/users/%s/evaluations/%s' % (user_id, id),
            headers = {'Authorization':'JCMXSorcbAAs8iVobTFGBxhX'},
            data = {'completed':True, 'kind':'JSON', 'content':json.dumps(content)})

        #from PIL import Image
        #Image.fromarray(result['mosaic_img']).save('mosaic.jpg', 'jpeg')
