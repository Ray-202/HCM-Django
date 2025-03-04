from django.contrib import admin

# Register your models here.
from .models import HRDocument

@admin.register(HRDocument)
class HRDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_at', 'processed')