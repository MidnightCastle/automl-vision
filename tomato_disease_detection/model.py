import os
from google.cloud import storage, automl
import pandas as pd
from PIL import Image
import base64
import json



class DiseaseDetection:

    project_id = 'delta-vector-163907'
    bucket_path = 'gs://tomato-disease-images'
    bucket_name = 'tomato-disease-images'
    dataset_id = 'ICN6984974170400489472'
    model_id = 'ICN5463677787581710336'
    model_display_name = 'tomato_disease_model_1'
    data_csv_uri = 'gs://tomato-disease-images/data.csv'
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/Users/dhiren/Desktop/Automl Vision/tomato_disease_detection/delta-vector-163907-70e238abb79e.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'./tomato_disease_detection/delta-vector-163907-70e238abb79e.json'


    def __init__(self):
        pass
    
    
    """ @staticmethod
    def get_project_id():
        return DiseaseDetection.project_id """
    
    """ @staticmethod
    def get_bucket_path():
        return DiseaseDetection.bucket_path """
    
    """ @staticmethod
    def get_bucket_name():
        return DiseaseDetection.bucket_name """

    """ @staticmethod
    def get_dataset_id():
        return DiseaseDetection.dataset_id """

    """ @staticmethod
    def get_model_id():
        return DiseaseDetection.model_id """
    
    """ @staticmethod
    def get_model_display_name():
        return DiseaseDetection.model_display_name """


    """ @staticmethod
    def get_blob_path_label():
        storage_client = storage.Client()
        gs_util_link = 'gs://tomato-disease-images'
        
        blobs = storage_client.list_blobs(DiseaseDetection.bucket_name)
        blob_dict = {'GS Path': [], 'Label': []}
    
        for blob in blobs:
            blob_path = gs_util_link + '/' + blob.name
            blob_label = blob.name.split('/')[0]
            blob_dict['GS Path'].append(blob_path)
            blob_dict['Label'].append(blob_label)
    
        blob_dict['GS Path'] = blob_dict['GS Path'][1:]
        blob_dict['Label'] = blob_dict['Label'][1:]

        return blob_dict """
    
    
    """ @staticmethod
    def upload_data_csv():
        blob_dict = DiseaseDetection.get_blob_path_label()
        df = pd.DataFrame(data=blob_dict)
        df.to_csv('data.csv', header=False, index=False)
        
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(DiseaseDetection.bucket_name)
        blob = bucket.blob('data.csv')

        blob.upload_from_filename('data.csv')
        data_csv_url = blob.public_url
        return data_csv_url """
    
    
    """ @staticmethod
    def create_dataset():
        display_name = 'tomato_disease'
        client = automl.AutoMlClient()

        project_location = client.location_path(DiseaseDetection.project_id, 'us-central1')
        metadata = automl.types.ImageClassificationDatasetMetadata(classification_type=automl.enums.ClassificationType.MULTICLASS)
        dataset = automl.types.Dataset(display_name= display_name, image_classification_dataset_metadata= metadata)
        response = client.create_dataset(project_location, dataset)

        created_dataset = response.result()
        dataset_id = created_dataset.name.split('/')[-1]
        dataset_dict = {'DatasetName': created_dataset.name, 'DatasetId': dataset_id}
        return dataset_dict """
    
    
    """ @staticmethod
    def import_dataset():
        path = DiseaseDetection.data_csv_uri
        
        client = automl.AutoMlClient()
        dataset_full_id = client.dataset_path(DiseaseDetection.project_id, 'us-central1', DiseaseDetection.dataset_id)
        input_uris = path.split(',')
        gcs_source = automl.types.GcsSource(input_uris=input_uris)
        input_config = automl.types.InputConfig(gcs_source=gcs_source)
        
        response = client.import_data(dataset_full_id, input_config)
        #print('Data imported: {}'.format(response.result()))
        #print(response.result())
        return True """

    """ @staticmethod
    def train_model():
        project_id = DiseaseDetection.project_id
        dataset_id = DiseaseDetection.dataset_id
        
        model_display_name = DiseaseDetection.model_display_name
        model_display_name = model_display_name.split('_')
        version = int(model_display_name[-1]) + 1
        model_display_name = model_display_name[:-1] + list(str(version))
        model_display_name = '_'.join(model_display_name)

        DiseaseDetection.model_display_name = model_display_name
        
        client = automl.AutoMlClient()

        project_location = client.location_path(project_id, 'us-central1')
        metadata = automl.types.ImageClassificationModelMetadata(train_budget_milli_node_hours = 24000)
        model = automl.types.Model(display_name=DiseaseDetection.model_display_name, dataset_id=dataset_id, image_classification_model_metadata=metadata)
        response = client.create_model(project_location, model)
        
        return True """
    

    """ @staticmethod
    def get_model_evaluation():

        client = automl.AutoMlClient()
        model_full_id = client.model_path(DiseaseDetection.project_id, 'us-central1', DiseaseDetection.model_id)

        model_evaluations = []
        
        for evaluation in client.list_model_evaluations(model_full_id, ''):
            details = {}

            details['EvaluationName'] = evaluation.name
            details['AnnotationSpecId'] = evaluation.annotation_spec_id
            details['CreateTimeSeconds'] = evaluation.create_time.seconds
            details['CreateTimeNanos'] = evaluation.create_time.nanos / 1e9
            details['EvaluationExampleCount'] = evaluation.evaluated_example_count
            details['ClassificationModelEvaluationMetrics'] = evaluation.classification_evaluation_metrics

            model_evaluations.append(details)
        
        return model_evaluations """

    """ @staticmethod
    def deploy_model():
        client = automl.AutoMlClient()
        model_full_id = client.model_path(DiseaseDetection.project_id, 'us-central1', DiseaseDetection.model_id)
        metadata = automl.types.ImageClassificationModelDeploymentMetadata(node_count=1)
        response = client.deploy_model(model_full_id, image_classification_model_deployment_metadata=metadata)
        return True """


    """ @staticmethod
    def get_deployed_model_details():
        project_id = DiseaseDetection.project_id
        client = automl.AutoMlClient()
        project_location = client.location_path(project_id, 'us-central1')
        response = client.list_models(project_location, "")

        models = []

        for model in response:
            
            details = {}

            if (model.deployment_state == automl.enums.Model.DeploymentState.DEPLOYED):
                deployment_state = "deployed"
            else:
                deployment_state = "undeployed"

            details['ModelName'] = model.name
            details['ModelId'] = model.name.split('/')[-1]
            details['ModelDisplayName'] = model.display_name
            details['ModelCreateTimeSeconds'] = model.create_time.seconds
            details['ModelCreateTimeNanos'] = model.create_time.nanos
            details['ModelDeploymentState'] = deployment_state

            models.append(details)
        
        return models """

    
    """ @staticmethod
    def undeploy_model(model_id):
        project_id = DiseaseDetection.project_id
        model_id = model_id
        client = automl.AutoMlClient()
        model_full_id = client.model_path(project_id, "us-central1", model_id)
        response = client.undeploy_model(model_full_id)

        return True """

    
    @staticmethod
    def model_predict(img_base64):
        project_id = DiseaseDetection.project_id
        model_id = DiseaseDetection.model_id
        prediction_client = automl.PredictionServiceClient()
        model_full_id = prediction_client.model_path(project_id, 'us-central1', model_id)
        
        img_b64 = img_base64
        img1 = base64.b64decode(img_b64)
        content = img1
        
        image = automl.types.Image(image_bytes=content)
        payload = automl.types.ExamplePayload(image=image)
        params = {'score_threshold' : '0.5'}
        
        response = prediction_client.predict(model_full_id, payload, params)
        
        prediction_result = []

        for result in response.payload:
            details = {}
            details['PredictedClassName'] = result.display_name
            details['PredictedClassScore'] = result.classification.score
            prediction_result.append(details)

        return prediction_result
    
    
    
    """ @staticmethod
    def get_blob_max_name(label):
        bucket_name = DiseaseDetection.bucket_name
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=label))
        blob_names = [ int(b.name.split('/')[-1].split('.')[0]) for b in blobs]

        return max(blob_names) """


    """ @staticmethod
    def upload_image_for_retraining(label, img_b64):
        bucket_name = DiseaseDetection.bucket_name

        max_name = DiseaseDetection.get_blob_max_name(label)
        max_name += 1
        max_name = str(max_name)+'.jpg'
        
        img1 = base64.b64decode(img_b64)

        with open(max_name, 'wb') as f:
            f.write(img1)

        img2 = Image.open(max_name)
        img2 = img2.resize((256,256))
        img2.save(max_name)
        #img1.show()

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        gcs_filename = label + '/' + max_name
        blob = bucket.blob(gcs_filename)
        #print(gcs_filename)
        with open(max_name, 'rb') as f:
            blob.upload_from_file(f)
            #print('image file uploaded')

        os.remove(max_name)

        return True """


