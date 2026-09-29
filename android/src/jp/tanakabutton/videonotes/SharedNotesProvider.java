package jp.tanakabutton.videonotes;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileNotFoundException;

/** Read-only, per-URI-granted access to files created solely for sharing. */
public final class SharedNotesProvider extends ContentProvider {
    @Override public boolean onCreate() { return true; }
    private File resolve(Uri uri) throws FileNotFoundException {
        String name = uri.getLastPathSegment();
        if (!"content".equals(uri.getScheme()) || !(getContext().getPackageName()+".sharednotes").equals(uri.getAuthority()) || uri.getPathSegments().size()!=1 || name==null || !name.matches("[a-f0-9-]{36}\\.txt")) throw new FileNotFoundException("Invalid shared note");
        File file = new File(new File(getContext().getCacheDir(), "shared-notes"), name);
        if (!file.isFile()) throw new FileNotFoundException("Shared note expired");
        return file;
    }
    @Override public String getType(Uri uri) { return "text/plain"; }
    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (!"r".equals(mode)) throw new FileNotFoundException("Read only");
        return ParcelFileDescriptor.open(resolve(uri), ParcelFileDescriptor.MODE_READ_ONLY);
    }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        try {
            File file=resolve(uri);
            String[] columns=projection==null?new String[]{OpenableColumns.DISPLAY_NAME,OpenableColumns.SIZE}:projection;
            MatrixCursor cursor=new MatrixCursor(columns,1);
            Object[] values=new Object[columns.length];
            for(int i=0;i<columns.length;i++) {
                if(OpenableColumns.DISPLAY_NAME.equals(columns[i])) values[i]="video-notes.txt";
                else if(OpenableColumns.SIZE.equals(columns[i])) values[i]=file.length();
            }
            cursor.addRow(values); return cursor;
        } catch(FileNotFoundException e) { return null; }
    }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException("Read only"); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException("Read only"); }
    @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException("Read only"); }
}
