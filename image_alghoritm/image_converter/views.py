import os

from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from image_converter.image_controller import handle_image, startRecognition


def show_image(request):
    return render(request, "index.html")


@csrf_exempt
def return_handled_image(request):
    function_name = request.POST.get('method')
    image_name = request.POST.get('image_name')
    image_path = os.path.join('image_alghoritm', 'static', 'image_alghoritm', image_name)
    handled_path = handle_image(image_path, function_name, request)
    return HttpResponse(os.path.split(handled_path)[1])

@csrf_exempt
def return_recognized_image(request):
    handled_path =startRecognition()
    return HttpResponse(handled_path)


@csrf_exempt
def upload_file(request):
    image_file = request.FILES.get('image')
    path = os.path.join('image_alghoritm', 'static', 'image_alghoritm', image_file.name)
    with open(path, 'wb+') as destination:
        for chunk in image_file.chunks():
            destination.write(chunk)
    return render(request, 'index.html')