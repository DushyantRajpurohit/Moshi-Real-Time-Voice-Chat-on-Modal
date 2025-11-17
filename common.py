import modal

# Initialize a Modal Application object.
# This object is the entry point for defining and deploying all functions,
# classes, and web endpoints related to the 'testproject'.
app = modal.App('testproject')
"""
The Modal App object for the 'testproject'.

This object registers all Modal functions, images, and volumes used
by the application.
"""