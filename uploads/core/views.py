from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse

from uploads.core.models import Document
from uploads.core.forms import DocumentForm

import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import tempfile

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content = f.read().decode()

        data = io.StringIO(file_content)
        df = pd.read_csv(data, sep='\t', index_col=0)
        #expression_df = pd.read_csv(filename, sep='\t', index_col=0)  
        heatmap_html = get_heatmap_html(df)
        plot_pca = get_plot_pca(df)

        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content,
            'heatmap_html': heatmap_html,
            'plot_pca' : plot_pca,
        })
    return render(request, 'core/simple_upload.html')

def get_heatmap_html(expression_df):
    fig, ax = plt.subplots(figsize=(20, 30))
    sns.heatmap(expression_df, cmap="YlGnBu", square=True, annot=False, ax=ax, cbar=False)
    heatmap_buffer = io.BytesIO()
    plt.savefig(heatmap_buffer, format='png')
    heatmap_image = heatmap_buffer.getvalue()
    heatmap_buffer.close()
    heatmap_image_base64 = base64.b64encode(heatmap_image).decode('utf-8')
    heatmap_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h2>Graph</h2><img src="data:image/png;base64,{heatmap_image_base64}"/></div>'    
    return heatmap_html

def Venn(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'core/venn.html')

def PCA_plot(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content = f.read().decode()

        data = io.StringIO(file_content)
        df = pd.read_csv(data, sep='\t', index_col=0)
        plot_pca = get_plot_pca(df)

        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content,
            'plot_pca' : plot_pca,
        })
    return render(request, 'core/simple_upload.html')

from sklearn.decomposition import PCA

def get_plot_pca(expression_df):
     # Perform PCA on the input data.
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(expression_df)

    # Create a scatter plot using matplotlib.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_results[:, 0], pca_results[:, 1])

    # Save the scatter plot as a PNG image.
    scatter_buffer = io.BytesIO()
    plt.savefig(scatter_buffer, format='png')
    scatter_image = scatter_buffer.getvalue()
    scatter_buffer.close()

    # Encode the PNG image as base64.
    scatter_image_base64 = base64.b64encode(scatter_image).decode('utf-8')

    # Construct the HTML for the scatter plot.
    #scatter_plot_html = f'<img src="data:image/png;base64,{scatter_image_base64}"/>'
    scatter_plot_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h2>Graph</h2><img src="data:image/png;base64,{scatter_image_base64}"/></div>'   

    return scatter_plot_html

def data_filter(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content = f.read().decode()

        file_content_10 = pd.read_csv(io.StringIO(file_content), sep='\t').head(10)
        data = io.StringIO(file_content)
        df = pd.read_csv(data, sep='\t', index_col=0)

        # logFC, FDR 값으로 필터링
        logFC_threshold = 1.0
        FDR_threshold = 0.05

        filtered_df = df[(df['logFC'] > logFC_threshold) & (df['FDR'] < FDR_threshold)]
        filtered_df_10 = filtered_df.head(10)

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            filtered_df.to_csv(temp.name, index=False)

        download_link = temp.name


        return render(request, 'core/filtering.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content_10,
            'filtered_df' : filtered_df_10,
            'download_link': download_link,
        })

    return render(request, 'core/filtering.html')


def download_filtered(request):
    filepath = request.GET.get('file')
    with open(filepath, 'rb') as f:
        response = HttpResponse(f, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="filtered_data.csv"'
        return response


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
