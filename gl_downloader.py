import gdown
import os

def download_from_gdrive(file_links: list, output_dir:str = 'data/raw'):
    """
    Download files from Google Drive links to specified directory and rename as CSV
    
    Args:
        file_links (list): List of Google Drive file links
        output_dir (str): Directory to save downloaded files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    file_names = ['census_income_learn.csv', 'census_income_test.csv']
    for i,link in enumerate(file_links):
        try:
            # Extract file ID from link
            file_id = link.split('/')[-2]
            
            # Create direct download URL
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Download file with custom output name
            print(file_names[i])
            output_path = os.path.join(output_dir, file_names[i])
            output = gdown.download(url, output=output_path, quiet=False)
            
            if output:
                print(f'Successfully downloaded file to {output}')
            else:
                print(f'Failed to download from {link}')
                
        except Exception as e:
            print(f'Error downloading from {link}: {str(e)}')