import hashlib
import re

import pytest

from ebim_phase2 import download


def test_parallel_ranges_preserve_partial_bytes_and_verify_assembled_hash(tmp_path,monkeypatch):
    payload=bytes(range(256))*512
    digest=hashlib.sha256(payload).hexdigest()
    target=tmp_path/'video.mp4'
    target.with_name('video.mp4.partial').write_bytes(payload[:10000])
    calls=[]
    class Response:
        status_code=206
        def __init__(self,start,end):
            self.start,self.end=start,end
            self.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):
            for offset in range(self.start,self.end+1,997):yield payload[offset:min(offset+997,self.end+1)]
    class Session:
        def get(self,url,headers,**kwargs):
            assert url=='https://huggingface.co/datasets/official/release/resolve/pinned/video.mp4'
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',headers['Range']).groups())
            calls.append((start,end));return Response(start,end)
    monkeypatch.setattr(download,'session',lambda:Session())
    item={'path':'video.mp4','size':len(payload),'lfs':{'oid':digest}}
    record=download.fetch_file_ranges('official/release','pinned',item,target,workers=3,chunk_bytes=8192)
    assert target.read_bytes()==payload and record['sha256']==digest
    assert min(start for start,end in calls)==10000
    assert not target.with_name('video.mp4.ranges').exists()
    calls.clear()
    download.fetch_file_ranges('official/release','pinned',item,target,workers=3,chunk_bytes=8192)
    assert calls==[]


def test_wrong_range_response_is_never_assembled(tmp_path,monkeypatch):
    class Response:
        status_code=200;headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
    class Session:
        def get(self,*args,**kwargs):return Response()
    monkeypatch.setattr(download,'session',lambda:Session())
    monkeypatch.setattr(download.time,'sleep',lambda _:None)
    target=tmp_path/'model.safetensors'
    with pytest.raises(ValueError,match='exact model byte range'):
        download.fetch_file_ranges('official/model','pinned',{'path':target.name,'size':10,'lfs':{'oid':'0'*64}},target,
                                   repo_type='model')
    assert not target.exists()
