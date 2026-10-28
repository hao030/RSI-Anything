import json

def feedback_text(feedback):
    return json.dumps([{'test_id': r['test_id'], 'status': r['status'], 'actual': r['actual'][:1024], 'stderr': r['stderr'][:2048], 'seconds': round(r['seconds'], 4)} for r in feedback['results']], ensure_ascii=False)
