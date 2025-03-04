

"""
URL configuration for chatbotHR project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from HCM_Chatbotapp.views import chatbot_view, chatbot_interface, whatsapp

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chatbot/', chatbot_view, name='chatbot'),
    path('', whatsapp, name='whatsapp'),
    path('chatbot_interface/', chatbot_interface, name='chat_interface'),
    #path('whatsapp/webhook/', whatsapp_webhook, name='whatsapp_webhook')
]

