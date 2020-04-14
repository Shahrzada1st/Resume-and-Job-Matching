#import os
import numpy as np
from io import StringIO

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams


os.chdir(r"C:\Users\khans\Insight\My code")

def pdfparser(data):

    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # removed from the line above: , codec=codec
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()

    return data


def resumeConvertor(resumename):
    with open('resumeconverted.txt','w') as f:
        f.write(pdfparser(str(resumename)))

    with open('resumeconverted.txt','r') as f:
        resume = f.read()

    return resume
