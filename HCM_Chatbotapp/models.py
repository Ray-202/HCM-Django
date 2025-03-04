from django.db import models

# Create your models here.
class HRDocument(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='pdfs/')  # store the PDF in media/pdfs/
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)  # mark if we've already embedded it

    def __str__(self):
        return self.title

class Department(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return self.name

class Employee(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    employee_id = models.CharField(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    department = models.ForeignKey(Department, on_delete=models.CASCADE,
                                   related_name='employees')
    job_title = models.CharField(max_length=100)
    def __str__(self):
        return f"{self.employee_id} - {self.first_name} {self.last_name}"