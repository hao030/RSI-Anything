"""Real Chromium executes the unmodified official task HTML; only timeout is overridden."""
import functools
import http.server
import os
import threading
import time
from .common import ROOT, correctness
from .settings import ASSETS

# Use Playwright's normal browser installation or the caller's explicit override.
from playwright.sync_api import sync_playwright

class EpisodeTerminated(RuntimeError):
    """An episode ended before the requested action was applied."""
    def __init__(self, observation):
        super().__init__('Cannot act after termination')
        self.observation = observation

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

class GUIEnv:
    def __init__(self, max_steps=8):
        directory = ASSETS
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1',0), functools.partial(QuietHandler,directory=str(directory)))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.pw=sync_playwright().start()
        self.browser=self.pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        self.page=self.browser.new_page(viewport={'width':500,'height':400})
        self.page.goto(f'http://127.0.0.1:{self.server.server_port}/miniwob/click-checkboxes.html',wait_until='load')
        self.max_steps=max_steps

    def reset(self, seed):
        self.steps=0
        self.submitted=False
        self.started=time.monotonic()
        self.page.evaluate('''seed=>{Math.seedrandom(String(seed));core.EPISODE_MAX_TIME=120000;core.startEpisodeReal();}''',seed)
        self.page.wait_for_function("()=>!WOB_DONE_GLOBAL && document.querySelectorAll('#boxes label').length>0")
        return self.observe()

    def observe(self):
        obs=self.page.evaluate('''()=>({instruction:document.querySelector('#query').innerText,
          elements:Array.from(document.querySelectorAll('#boxes label')).map((e,i)=>({slot:i,ref:e.querySelector('input').id,label:e.innerText.trim(),checked:e.querySelector('input').checked})),
          done:Boolean(WOB_DONE_GLOBAL),raw_reward:Number(WOB_RAW_REWARD_GLOBAL),reward_reason:WOB_REWARD_REASON})''')
        obs.update(remaining_actions=max(0,self.max_steps-self.steps),submitted=self.submitted)
        elapsed = time.monotonic()-self.started
        if self.steps>=self.max_steps or elapsed>=120:
            obs['done']=True
            obs['budget_exhausted']=not self.submitted
            if not self.submitted:
                obs['termination_reason']='timeout' if elapsed>=120 else 'action_budget'
                obs['elapsed_seconds']=elapsed
        return obs

    def step(self, action):
        before=self.observe()
        if before['done']:
            raise EpisodeTerminated(before)
        started=time.monotonic()
        if action['kind']=='click':
            slot=action['slot']
            if not isinstance(slot,int) or not 0<=slot<len(before['elements']):raise ValueError('Invalid slot')
            self.page.locator('#'+before['elements'][slot]['ref']).click(force=True)
        elif action['kind']=='submit':
            self.submitted=True
            self.page.locator('#subbtn').click(force=True)
        else:raise ValueError('Invalid action')
        self.steps+=1
        return self.observe(),time.monotonic()-started

    def replay(self,seed,actions,expected):
        obs=self.reset(seed)
        for action in actions:
            obs,_=self.step(action)
        # Wall time and reward are deliberately excluded; visible state must match exactly.
        if any(obs[k]!=expected[k] for k in ['instruction','elements','done','submitted','remaining_actions']):
            raise RuntimeError('Browser replay failed exact visible-state check')
        return obs

    def screenshot(self,path):
        self.page.screenshot(path=str(path))

    def close(self):
        self.browser.close()
        self.pw.stop()
        self.server.shutdown()
        self.server.server_close()

def rule_action(obs):
    from .common import goal_bits
    for e,g in zip(obs['elements'],goal_bits(obs)):
        if e['checked']!=g:return {'kind':'click','slot':e['slot']}
    return {'kind':'submit'}
