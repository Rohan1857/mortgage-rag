import os
import io
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from typing import List, Dict, Optional

class DriveManager:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/drive']
        
        # Look for the OAuth credentials file (preferring env var)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if env_path:
            if os.path.isabs(env_path):
                self.creds_path = env_path
            else:
                self.creds_path = os.path.join(script_dir, env_path)
        else:
            self.creds_path = os.path.join(script_dir, "cred_drive.json")
            
        token_path = os.path.join(script_dir, "token.json")

        if not os.path.exists(self.creds_path):
            raise FileNotFoundError(f"OAuth credentials not found at {self.creds_path}. Download your OAuth Client ID JSON from Google Cloud Console and ensure it matches the path in your .env or is named 'cred_drive.json'.")

        creds = None
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.scopes)
            
        # If there are no valid credentials, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, self.scopes)
                # This will open a browser window for you to log in
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run so you don't have to log in every time
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        self.service = build('drive', 'v3', credentials=creds)
        
        self.pdf_folder_id = os.getenv("DRIVE_PDF_FOLDER_ID")
        self.dataset_folder_id = os.getenv("DRIVE_DATASET_FOLDER_ID")
        self.results_root_id = os.getenv("DRIVE_RESULTS_ROOT_ID")

    def list_pdfs(self) -> List[Dict[str, str]]:
        """Lists files in the PDF folder."""
        query = f"'{self.pdf_folder_id}' in parents and mimeType = 'application/pdf' and trashed = false"
        results = self.service.files().list(
            q=query, 
            fields="files(id, name, appProperties)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        return results.get('files', [])

    def is_processed_by_user(self, file_id: str, user_id: str) -> bool:
        """Checks if a file has been processed by the given user using appProperties."""
        file_metadata = self.service.files().get(
            fileId=file_id, fields="appProperties", supportsAllDrives=True
        ).execute()
        
        app_props = file_metadata.get('appProperties', {})
        key = f"processed_by_{user_id}"
        return app_props.get(key) == "true"

    def mark_as_processed(self, file_id: str, user_id: str):
        """Updates appProperties to mark the file as processed by the user."""
        key = f"processed_by_{user_id}"
        body = {
            "appProperties": {
                key: "true"
            }
        }
        self.service.files().update(fileId=file_id, body=body, supportsAllDrives=True).execute()

    def report_exists(self, filename: str, user_id: str) -> bool:
        """Checks if a report file already exists in the user's folder."""
        folder_id = self.get_or_create_user_folder(user_id)
        query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
        results = self.service.files().list(
            q=query, 
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        return len(results.get('files', [])) > 0

    def download_file(self, file_id: str, local_name: Optional[str] = None) -> str:
        """Downloads a file from Drive and returns the local path."""
        if not local_name:
            metadata = self.service.files().get(fileId=file_id, fields="name", supportsAllDrives=True).execute()
            local_name = metadata['name']
        
        # Create temp dir if not exists (relative to script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, "temp_eval")
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, local_name)
        
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with io.FileIO(local_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
        
        return local_path

    def download_corresponding_qa(self, pdf_name: str) -> Optional[str]:
        """Finds and downloads the _qa.json file from the dataset folder."""
        base_name = os.path.splitext(pdf_name)[0]
        qa_name = f"{base_name}_qa.json"
        
        query = f"'{self.dataset_folder_id}' in parents and name = '{qa_name}' and trashed = false"
        results = self.service.files().list(
            q=query, 
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get('files', [])
        
        if not files:
            # Fallback check for just .json
            qa_name = f"{base_name}.json"
            query = f"'{self.dataset_folder_id}' in parents and name = '{qa_name}' and trashed = false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            files = results.get('files', [])

        if not files:
            print(f"Warning: No QA file found for {pdf_name} in dataset folder.")
            return None
            
        return self.download_file(files[0]['id'], local_name=qa_name)

    def get_or_create_user_folder(self, user_id: str) -> str:
        """Finds or creates a folder named {user_id} inside the Results root."""
        query = f"'{self.results_root_id}' in parents and name = '{user_id}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = self.service.files().list(
            q=query, 
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        
        # Create it
        file_metadata = {
            'name': user_id,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [self.results_root_id]
        }
        folder = self.service.files().create(
            body=file_metadata, 
            fields='id',
            supportsAllDrives=True
        ).execute()
        return folder.get('id')

    def upload_to_user_folder(self, content: str, filename: str, user_id: str, mime_type: str = 'text/plain'):
        """Uploads a file to the user-specific result folder."""
        target_folder_id = self.get_or_create_user_folder(user_id)
        
        file_metadata = {
            'name': filename,
            'parents': [target_folder_id]
        }
        
        # Use BytesIO to avoid temp file locks on Windows
        fh = io.BytesIO(content.encode('utf-8'))
        media = MediaIoBaseUpload(fh, mimetype=mime_type, resumable=True)
        
        try:
            self.service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields='id',
                supportsAllDrives=True
            ).execute()
        except Exception as e:
            err_msg = str(e).lower()
            if "storagequotaexceeded" in err_msg:
                print(f"❌ Drive Error: Storage quota exceeded. Please check your Google Drive storage (5TB limit).")
            raise e
