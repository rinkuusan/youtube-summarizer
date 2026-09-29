package jp.tanakabutton.videonotes;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.LinearLayout;
import android.widget.Toast;
import java.util.LinkedHashSet;
import org.json.JSONArray;

public final class MainActivity extends Activity {
    private static final String HOME = "https://youtube-summarizer-of5v.onrender.com/__android__/index.html";
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final LinkedHashSet<String> pending = new LinkedHashSet<>();
    private WebView web;
    private ClipboardManager clipboard;
    private boolean loaded = false, focused = false, importing = false;
    private boolean sharedIntentPending = false;
    private String lastClipboard = "";
    private ValueCallback<Uri[]> fileCallback;
    private String pendingSave;
    private final ClipboardManager.OnPrimaryClipChangedListener clipListener = () -> { if (focused) readClipboard(false); };
    private final Runnable deliver = () -> deliverPending();

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        if (android.os.Build.VERSION.SDK_INT >= 33) getOnBackInvokedDispatcher().registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT, () -> moveTaskToBack(true));
        getWindow().setStatusBarColor(Color.rgb(11,11,16));
        getWindow().setNavigationBarColor(Color.rgb(11,11,16));
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setBackgroundColor(Color.rgb(11,11,16));
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(android.view.WindowInsets.Type.systemBars() | android.view.WindowInsets.Type.ime());
                view.setPadding(bars.left,bars.top,bars.right,bars.bottom);
            } else view.setPadding(insets.getSystemWindowInsetLeft(),insets.getSystemWindowInsetTop(),insets.getSystemWindowInsetRight(),insets.getSystemWindowInsetBottom());
            return insets;
        });
        web = new WebView(this); web.setBackgroundColor(Color.rgb(11,11,16));
        root.addView(web,new LinearLayout.LayoutParams(-1,-1)); setContentView(root);
        WebSettings settings = web.getSettings(); settings.setJavaScriptEnabled(true); settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false); settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setSupportMultipleWindows(false); settings.setJavaScriptCanOpenWindowsAutomatically(false);
        web.addJavascriptInterface(new ClipboardBridge(), "AndroidClipboard");
        web.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest req) {
                if (HOME.equals(req.getUrl().toString())) {
                    try { return new WebResourceResponse("text/html","UTF-8",getAssets().open("index.html")); }
                    catch (Exception e) { return new WebResourceResponse("text/plain","UTF-8",new java.io.ByteArrayInputStream("Page unavailable".getBytes())); }
                }
                if (req.isForMainFrame()) return new WebResourceResponse("text/plain","UTF-8",new java.io.ByteArrayInputStream(new byte[0]));
                return null;
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                String url = req.getUrl().toString();
                if (HOME.equals(url)) return false;
                if (req.isForMainFrame() && ("https".equals(req.getUrl().getScheme()) || "http".equals(req.getUrl().getScheme()))) {
                    try { startActivity(new Intent(Intent.ACTION_VIEW, req.getUrl())); } catch (Exception ignored) {}
                }
                return true;
            }
            @Override public void onPageStarted(WebView view,String url,android.graphics.Bitmap icon) { loaded = false; }
            @Override public void onPageFinished(WebView view,String url) { loaded = HOME.equals(url); if (loaded) { deliverPending(); if (focused) readClipboard(false); } }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent pick = new Intent(Intent.ACTION_OPEN_DOCUMENT); pick.addCategory(Intent.CATEGORY_OPENABLE);
                pick.setType("*/*"); pick.putExtra(Intent.EXTRA_MIME_TYPES,new String[]{"audio/*","video/*"});
                try { startActivityForResult(pick, 41); } catch (Exception e) { fileCallback.onReceiveValue(null); fileCallback = null; }
                return true;
            }
        });
        clipboard = (ClipboardManager)getSystemService(CLIPBOARD_SERVICE);
        clipboard.addPrimaryClipChangedListener(clipListener);
        if (state != null) { java.util.ArrayList<String> saved = state.getStringArrayList("pending"); if (saved != null) pending.addAll(saved); lastClipboard = state.getString("lastClipboard", ""); }
        if (state == null) receiveIntent(getIntent());
        web.loadUrl(HOME);
    }
    @Override public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus); focused = hasFocus;
        if (hasFocus && clipboard != null) { handler.postDelayed(() -> { if (focused) { readClipboard(false); deliverPending(); } }, 200); }
    }
    private void receiveIntent(Intent intent) {
        if (intent != null && Intent.ACTION_SEND.equals(intent.getAction()) && "text/plain".equals(intent.getType())) {
            CharSequence text = intent.getCharSequenceExtra(Intent.EXTRA_TEXT);
            if (text != null) { LinkedHashSet<String> urls = YouTubeUrls.extract(text.toString()); pending.addAll(urls); sharedIntentPending = !urls.isEmpty(); if (urls.isEmpty()) Toast.makeText(this,"共有されたテキストにYouTubeの動画URLがありません",Toast.LENGTH_LONG).show(); }
        }
    }
    @Override protected void onNewIntent(Intent intent) { super.onNewIntent(intent); setIntent(intent); receiveIntent(intent); deliverPending(); }
    private void readClipboard(boolean force) {
        if (!focused) return;
        try {
            boolean skipSharedClipboard = sharedIntentPending && !force; sharedIntentPending = false;
            ClipData clip = clipboard.getPrimaryClip();
            if (clip == null || clip.getItemCount() == 0) return;
            // Never coerce URIs or arbitrary clipboard items into content reads.
            CharSequence value = clip.getItemAt(0).getText(); if (value == null) return;
            LinkedHashSet<String> urls = YouTubeUrls.extract(value.toString());
            if (urls.isEmpty()) { if (force) Toast.makeText(this,"コピー済みのYouTube URLがありません",Toast.LENGTH_SHORT).show(); return; }
            String fingerprint = urls.toString();
            if (skipSharedClipboard) { lastClipboard = fingerprint; return; }
            if (!force && fingerprint.equals(lastClipboard)) return;
            lastClipboard = fingerprint; pending.addAll(urls); deliverPending();
        } catch (SecurityException ignored) {}
    }
    private void deliverPending() {
        handler.removeCallbacks(deliver);
        if (!loaded || !focused || pending.isEmpty() || importing || !HOME.equals(web.getUrl())) return;
        final LinkedHashSet<String> snapshot = new LinkedHashSet<>(pending);
        importing = true;
        web.evaluateJavascript("window.importYouTubeUrls("+new JSONArray(snapshot).toString()+")", value -> {
            importing = false;
            if ("true".equals(value)) pending.removeAll(snapshot);
            else if (focused) handler.postDelayed(deliver,1000);
        });
    }
    private final class ClipboardBridge {
        @JavascriptInterface public void share(String text) {
            if (text == null || text.trim().isEmpty()) return;
            handler.post(() -> {
                if (!focused || !loaded || !HOME.equals(web.getUrl())) return;
                new Thread(() -> {
                    try {
                        Intent send = new Intent(Intent.ACTION_SEND).setType("text/plain");
                        send.putExtra(Intent.EXTRA_SUBJECT, "動画ノート");
                        // Large Binder extras can fail: attach the complete UTF-8 text instead.
                        if (text.length() <= 60000) {
                            send.putExtra(Intent.EXTRA_TEXT, text);
                        } else {
                            java.io.File dir = new java.io.File(getCacheDir(), "shared-notes");
                            if (!dir.isDirectory() && !dir.mkdirs()) throw new java.io.IOException("Cannot create share directory");
                            String name = java.util.UUID.randomUUID().toString() + ".txt";
                            try (java.io.FileOutputStream out = new java.io.FileOutputStream(new java.io.File(dir, name))) {
                                out.write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                            }
                            Uri uri = new Uri.Builder().scheme("content").authority(getPackageName() + ".sharednotes").appendPath(name).build();
                            send.putExtra(Intent.EXTRA_STREAM, uri);
                            send.putExtra(Intent.EXTRA_TEXT, "動画ノートの全文をTXTファイルに添付しました。");
                            send.setClipData(ClipData.newUri(getContentResolver(), "動画ノート全文", uri));
                            send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                        }
                        handler.post(() -> {
                            if (isFinishing() || isDestroyed() || !focused || !loaded || !HOME.equals(web.getUrl())) return;
                            try { startActivity(Intent.createChooser(send, "動画ノートを共有")); }
                            catch (Exception e) { Toast.makeText(MainActivity.this,"共有先を開けませんでした。コピーまたはTXT保存を使ってください",Toast.LENGTH_LONG).show(); }
                        });
                    } catch (Exception e) {
                        handler.post(() -> Toast.makeText(MainActivity.this,"共有の準備に失敗しました。TXT保存を使ってください",Toast.LENGTH_LONG).show());
                    }
                }, "share-notes").start();
            });
        }
        @JavascriptInterface public void save(String name, String text) {
            if (text == null || text.length() > 3000000) return;
            handler.post(() -> {
                if (!focused || !loaded || !HOME.equals(web.getUrl()) || pendingSave != null) return;
                pendingSave = text;
                Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("text/plain");
                intent.putExtra(Intent.EXTRA_TITLE, name == null ? "transcript.txt" : name.replaceAll("[^a-zA-Z0-9._-]", "_"));
                try { startActivityForResult(intent, 42); }
                catch (Exception e) { pendingSave = null; Toast.makeText(MainActivity.this,"保存先を開けませんでした",Toast.LENGTH_LONG).show(); }
            });
        }
        @JavascriptInterface public void copy(String text) {
            if (text == null || text.length() > 3000000) return;
            handler.post(() -> { if (focused && loaded && HOME.equals(web.getUrl())) { lastClipboard = YouTubeUrls.extract(text).toString(); clipboard.setPrimaryClip(ClipData.newPlainText("動画ノート",text)); } });
        }
        @JavascriptInterface public void paste() { handler.post(() -> readClipboard(true)); }
    }
    @Override protected void onActivityResult(int request,int result,Intent data) {
        super.onActivityResult(request,result,data);
        if (request == 42) {
            final String text = pendingSave; pendingSave = null;
            if (result == RESULT_OK && data != null && data.getData() != null && text != null) {
                final Uri uri = data.getData();
                new Thread(() -> {
                    try (java.io.OutputStream stream = getContentResolver().openOutputStream(uri)) {
                        if (stream == null) throw new java.io.IOException("No output stream");
                        stream.write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        handler.post(() -> Toast.makeText(this,"全文を保存しました",Toast.LENGTH_SHORT).show());
                    } catch (Exception e) { handler.post(() -> Toast.makeText(this,"保存に失敗しました。もう一度保存してください",Toast.LENGTH_LONG).show()); }
                }, "save-transcript").start();
            }
        }
        if (request == 41 && fileCallback != null) { fileCallback.onReceiveValue(result == RESULT_OK && data != null && data.getData() != null ? new Uri[]{data.getData()} : null); fileCallback = null; }
    }
    @Override protected void onSaveInstanceState(Bundle out) { out.putStringArrayList("pending",new java.util.ArrayList<>(pending)); out.putString("lastClipboard",lastClipboard); super.onSaveInstanceState(out); }
    @Override public void onBackPressed() { moveTaskToBack(true); }
    @Override protected void onPause() { focused = false; handler.removeCallbacks(deliver); super.onPause(); }
    @Override protected void onDestroy() { handler.removeCallbacksAndMessages(null); if (clipboard != null) clipboard.removePrimaryClipChangedListener(clipListener); if (fileCallback != null) fileCallback.onReceiveValue(null); web.removeJavascriptInterface("AndroidClipboard"); web.destroy(); super.onDestroy(); }
}
